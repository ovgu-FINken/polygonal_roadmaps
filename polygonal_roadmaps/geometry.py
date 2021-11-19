import numpy as np
from scipy.spatial import Voronoi
import shapely.geometry
from shapely.geometry import Polygon, Point, MultiLineString, LineString
import shapely.ops
import matplotlib.pyplot as plt
from dataclasses import dataclass
from skimage import io, measure
import yaml
import os
import networkx as nx

from polygonal_roadmaps import pathfinding


@dataclass
class NavNode:
    """class for keeping node data"""
    center: Point
    outer: Polygon
    inner: Polygon
    name: str

    def get_center_np(self) -> np.array:
        return np.array(self.center.xy)[:, 0]


@dataclass
class NavEdge:
    """ class for edges between nodes """
    borderLine: MultiLineString
    connection: MultiLineString
    borderPoly: Polygon


def create_graph(generator_points: np.array,
                 working_area_x=(0.0, 1.0),
                 working_area_y=(0.0, 1.0),
                 offset: float = 0.02,
                 occupied_space=None):
    limits = polygon_from_limits(working_area_x, working_area_y)
    if occupied_space is not None:
        limits = limits.difference(occupied_space)

    nodes = add_nodes_to_graph(generator_points, limits, offset)
    edges = add_edges_to_graph(nodes, offset)
    return gen_graph_nx(nodes, edges)


def add_nodes_to_graph(generator_points, limits, offset):
    # we add the outer points in order to get closed cells (not ending in infinity)
    padded_generators = np.vstack([generator_points, np.array([[0, 100], [0, -100], [-100, 0], [100, 0]])])
    vor = Voronoi(padded_generators)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    polys = [p for p in shapely.ops.polygonize(lines)]
    nodes = []
    for i, poly in enumerate(polys):
        x, y = None, None
        for p in generator_points:
            if shapely.geometry.Point(p).within(poly):
                x, y = p[0], p[1]
                break
        outer = poly.intersection(limits)

        # in case polygon is split by obstacles -- use largest polygon as shape
        outer = select_largest_poly(outer)
        inner = outer.buffer(-offset)
        inner = select_largest_poly(inner)
        if not inner.is_empty:
            nodes.append(NavNode(center=Point(x, y), outer=outer, inner=inner, name=f'N{i}'))
    return nodes


def add_edges_to_graph(nodes, offset):
    edges = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            x = shapely.ops.shared_paths(n1.outer.exterior, n2.outer.exterior)
            if x.is_empty:
                continue
            e = [a for a in x if not a.is_empty and a.length > 0]
            if len(e) > 0:
                if n1.inner is None or n2.inner is None or n1.inner.is_empty or n2.inner.is_empty \
                        or not n1.inner.is_valid or not n2.inner.is_valid:
                    continue

                bp = n1.outer.union(n2.outer).buffer(-offset).difference(n1.inner).difference(n2.inner)
                if bp.is_valid:
                    if bp.geometryType() == 'MultiPolygon':
                        x = None
                        for p in bp:
                            if p.touches(n1.inner) and p.touches(n2.inner):
                                x = p
                        bp = x
                if bp.touches(n1.inner) and bp.touches(n2.inner):
                    edges.append(
                        (
                            n1, n2, NavEdge(
                                borderLine=e[0],
                                connection=LineString([n1.center, n2.center]),
                                borderPoly=bp),
                            i, j
                        )
                    )
    return edges


def polygon_from_limits(working_area_x, working_area_y):
    return Polygon([
        (working_area_x[0], working_area_y[0]),
        (working_area_x[1], working_area_y[0]),
        (working_area_x[1], working_area_y[1]),
        (working_area_x[0], working_area_y[1])
    ])


def square_tiling(dist, working_area_x=(0.0, 1.0), working_area_y=(0.0, 1.0)):
    XX, YY = np.meshgrid(np.arange(working_area_x[0] + dist / 2, working_area_x[1], dist),
                         np.arange(working_area_y[0] + dist / 2, working_area_y[1], dist), indexing='xy')
    return np.dstack([XX, YY]).reshape(-1, 2)


def hexagon_tiling(dist, working_area_x=(0.0, 1.0), working_area_y=(0.0, 1.0)):
    dy = dist * np.sqrt(0.75)
    XX, YY = np.meshgrid(np.arange(working_area_x[0] + dist * 0.1, working_area_x[1] - dist / 2, dist),
                         np.arange(working_area_y[0] + dist / 2, working_area_y[1], dy), indexing='xy')
    XX[::2] += dist * 0.5
    return np.dstack([XX, YY]).reshape(-1, 2)


def random_tiling(n, working_area_x=(0.0, 1.0), working_area_y=(0.0, 1.0)):
    t = np.random.random((n, 2))
    t[:, 0] = t[:, 0] * (working_area_x[1] - working_area_x[0]) + working_area_x[0]
    t[:, 1] = t[:, 1] * (working_area_y[1] - working_area_y[0]) + working_area_y[0]
    return t


def read_map(info_file):
    with open(info_file, 'r') as stream:
        info = yaml.safe_load(stream)
    img_file = info['image']
    if img_file[0] not in ['/', '~']:
        img_file = os.path.join(os.path.dirname(info_file), img_file)
    img = io.imread(img_file).transpose()
    return img, info


def poly_from_path(g, path, eps=0.05):
    poly = [g.nodes()[p]['geometry'].inner.buffer(eps) for p in path]
    poly += [g.edges()[n1, n2]['geometry'].borderPoly.buffer(eps) for n1, n2 in zip(path[:-1], path[1:])]
    poly = shapely.ops.unary_union(poly).buffer(-eps).simplify(eps)
    return select_largest_poly(poly)


def path_from_positions(g, start, goal):
    sn = find_nearest_node(g, start)
    gn = find_nearest_node(g, goal)
    return pathfinding.spatial_astar(g, sn, gn)


def convert_coordinates(poly, resolution: float, oX: float, oY: float):
    poly = poly * resolution + np.array([oX + 0.025, oY + 1.45])
    poly[:, 1] *= -1
    return Polygon(poly)


def read_obstacles(file_name):
    img, info = read_map(file_name)
    thresh = 100
    contours = measure.find_contours(img, level=thresh)
    unclassified = contours
    obstacles, free = [], []

    # classify different levels of objects
    # holes need to be filled by opposite value
    while unclassified:
        for i, poly in enumerate(unclassified):
            if np.max(img[measure.grid_points_in_poly(img.shape, poly)]) <= thresh:
                img[measure.grid_points_in_poly(img.shape, poly)] = 200
                del unclassified[i]
                p = convert_coordinates(poly, info['resolution'], info['origin'][0], info['origin'][1])
                for f in free:
                    if p.contains(f):
                        p = p.difference(f)
                obstacles.append(p)
                break

            if np.min(img[measure.grid_points_in_poly(img.shape, poly)]) >= thresh:
                img[measure.grid_points_in_poly(img.shape, poly)] = 30
                del unclassified[i]
                p = convert_coordinates(poly, info['resolution'], info['origin'][0], info['origin'][1])
                for o in obstacles:
                    if p.contains(o):
                        p = p.difference(o)
                free.append(p)
                break

    return shapely.ops.unary_union(free), shapely.ops.unary_union(obstacles)


def gen_graph_nx(nodes, edges):
    g = nx.Graph()
    for i, n in enumerate(nodes):
        g.add_node(i, pos=(n.center.x, n.center.y), traversable=n.inner is not None, geometry=n)
    for le in edges:
        dist = le[0].center.distance(le[1].center)
        border_poly = le[2].borderPoly
        if border_poly is None or not border_poly.is_valid or border_poly.is_empty:
            traversable = False
        else:
            bigger_poly = border_poly.buffer(1e-8)
            traversable = bigger_poly.intersects(le[0].inner) and bigger_poly.intersects(le[1].inner)
        g.add_edge(le[-2], le[-1], geometry=le[2], dist=dist, traversable=traversable)
    return g


def drawGraph(g,
              inner=True,
              outer=True,
              center=True,
              connections=True,
              borderPolys=True,
              show=True,
              edge_weight=None,
              node_weight=None):
    wax = g.gp['wa']['working_area_x']
    way = g.gp['wa']['working_area_y']
    width = 6
    height = width * (way[1] - way[0]) / (wax[1] - wax[0])

    plt.figure(figsize=(width, height))
    plot_vertices(g, outer, center, node_weight)

    plot_edges(g, connections, borderPolys, edge_weight)
    if show:
        plt.show()


def plot_edges(g, connections, borderPolys, edge_weight):
    for ge in g.edges():
        e = g.ep['geometry'][ge]
        if connections:
            lw = None
            if edge_weight is not None:
                lw = 4.0 * g.ep[edge_weight][ge] / g.ep[edge_weight].a.max()
            plt.plot(*e.connection.xy, lw=lw, color='b', alpha=0.3)
        if borderPolys and e.borderPoly is not None and not e.borderPoly.is_empty:
            if e.borderPoly.geometryType() == 'MultiPolygon':
                for p in e.borderPoly:
                    plt.plot(*p.exterior.xy)
            else:
                plt.fill(*e.borderPoly.exterior.xy, alpha=0.1)


def plot_vertices(g, outer, center, node_weight):
    for v in g.iter_vertices():
        n = g.vp['geometry'][v]
        if n.inner and n.inner is not None:
            plt.plot(*n.inner.exterior.xy, ':', color='black')
        if outer:
            plt.plot(*n.outer.exterior.xy, '--', color='black')
        if center:
            s = 5
            if node_weight is not None:
                s *= 2.0 * g.vp[node_weight][v] / g.vp[node_weight].a.max()

            plt.plot(*n.center.xy, 'o', markersize=s, color='black')


def find_nearest_node(g, p):
    dist = [np.linalg.norm(np.array(p) - g.nodes()[n]['geometry'].get_center_np()) for n in g.nodes()]
    return np.argmin(dist)


def waypoints_through_poly(g, poly, start, goal, eps=0.01):
    """ compute a waypoints of a linear line segement path through a polygon

    """
    # coords will hold final waypoints
    coords = [start]
    # recompute start point if start is outside polygon
    start = point_on_border(g, poly, start)
    if start is not None:
        coords.append(start)

    ep = point_on_border(g, poly, goal)
    # if endpoint is outside poly, use endpoint on border and append goal
    if ep is not None:
        straight_path = compute_straight_path(g, poly, coords[-1], ep)
        coords += shorten_path(g, poly, straight_path)
        coords.append(goal)
    else:
        straight_path = compute_straight_path(g, poly, coords[-1], goal)
        coords += shorten_path(g, poly, straight_path)
    coords = remove_close_points(coords, eps=eps)
    return LineString(coords)


def remove_close_points(coords, eps=0.01):
    return LineString(coords).simplify(eps).coords


def shorten_path(g, poly, coords):
    shorter = list(reversed(shorten_recursive(g, poly, coords)))
    shortest = list(reversed(shorten_recursive(g, poly, shorter)))
    return shortest


def shorten_recursive(g, poly, coords, eps=0.01):
    if len(coords) < 3:
        return coords
    # if next segment can be dropped drop the segment
    if LineString([coords[0], coords[2]]).within(poly):
        coords = [coords[0]] + coords[2:]
        return shorten_recursive(g, poly, coords, eps=eps)

    if len(coords) < 4:
        return [coords[0]] + shorten_recursive(g, poly, coords[1:], eps=eps)

    straight_segment = compute_straight_path(g, poly, coords[1], coords[-1], eps=eps)
    # straight_segment = coords[1:]
    return [coords[0]] + shorten_recursive(g, poly, straight_segment, eps=eps)


def shorten_direction(g, poly, coords):
    # stop recursion: cannot reduce 2 point line
    if len(coords) < 3:
        return coords
    for i in range(1, len(coords) - 1):
        if LineString([coords[i - 1], coords[i + 1]]).within(poly):
            jm = i + 1
            for j in range(i + 2, len(coords)):
                if LineString([coords[i - 1], coords[j]]).within(poly):
                    jm = j
                else:
                    break
            ret = coords[:i]
            ret += shorten_direction(g, poly, coords[jm:])
            return ret
    # no reduction possible:
    return coords


def point_on_border(g, poly: Polygon, point: Point) -> Point:
    """
    compute a linestring to the edge of the polygon
    if the point is in the polygone:
        return none
    if the point is outside:
        return point on edge
    """
    point = Point(point)
    if point.within(poly):
        return None
    sn = find_nearest_node(g, point)
    inner = g.vp['geometry'][sn].inner
    d = inner.exterior.project(point)
    return inner.exterior.interpolate(d)


def compute_straight_path(g, poly, start, goal, eps=None):
    line = LineString([start, goal])
    if line.length < 0.01 or line.within(poly):
        return [start, goal]
    if not line.intersects(poly):
        print("need to snap line")
        line = shapely.ops.snap(line, poly, 0.01)
    assert(line.intersects(poly))
    inner_line = poly.intersection(line)
    if inner_line.geometryType() == 'LineString':
        inner_line = MultiLineString([inner_line])
    elif inner_line.geometryType() == 'Point':
        inner_line = MultiLineString([])
    elif inner_line.geometryType() == 'MultiLineString':
        pass
    else:
        # print(inner_line.wkt)
        pass
    result = [i for i in inner_line if i.geometryType() == 'LineString']
    outer_line = line - poly
    if outer_line.geometryType() == 'LineString':
        if outer_line.is_empty:
            outer_line = MultiLineString([])
        else:
            outer_line = MultiLineString([outer_line])
    for ls in outer_line:
        i0 = poly.exterior.project(Point(ls.coords[0]))
        i1 = poly.exterior.project(Point(ls.coords[-1]))
        outer_segment = shapely.ops.substring(poly.exterior, i0, i1)
        if outer_segment.geometryType() == 'LineString':
            outer_segment = MultiLineString([outer_segment])
        elif outer_segment.geometryType() == 'Point':
            continue
        result += [i for i in outer_segment if i.geometryType() == 'LineString']

    try:
        line = shapely.ops.linemerge(MultiLineString(result))
    except AssertionError:
        print("ASSERTION ERROR")
        print(result)
    if isinstance(line, shapely.geometry.base.BaseMultipartGeometry):
        line = list(line)
        for i, ls in enumerate(line):
            for j, l2 in enumerate(line):
                if i == j:
                    continue
                if Point(ls.coords[0]).almost_equals(Point(l2.coords[-1])):
                    c = list(ls.coords)
                    c[0] = l2.coords[-1]
                    line[i] = LineString(c)
                    break
        line = shapely.ops.linemerge(line)

    if isinstance(line, shapely.geometry.base.BaseMultipartGeometry):
        print("failed to merge multiline-string")
        print(line.wkt)
        print(result)
        print("inner:")
        print(inner_line)
        print("outer:")
        print(outer_line)
        if not len(line):
            return [start, goal]
        assert(len(line))
        coords = [line[0].coords[0]]
        for segment in line:
            if len(segment.coords) > 1:
                coords += list(segment.coords)[1:]
        return coords
    if eps:
        line = line.simplify(eps)
    return list(line.coords)


def select_largest_poly(poly):
    if poly.geometryType() == 'MultiPolygon':
        return sorted(poly, key=lambda p: p.area, reverse=True)[0]
    return poly

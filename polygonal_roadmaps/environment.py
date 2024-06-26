import warnings
import networkx as nx
import logging
import pandas as pd
import numpy as np
from polygonal_roadmaps import geometry

from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd


@dataclass(frozen=True)
class PlanningProblemParameters:
    """
    Parameters for the planning problem, that are not specific for an individual planner.
    """
    conflict_horizon: int | None = None
    k_robustness: int = 1
    weight_name: str = "dist"
    wait_action_cost: float = 1.0001
    pad_path: bool = False
    step_num: int | None = None
    lifelong: bool = False


def remove_edge_if_exists(g: nx.Graph, u, v) -> None:
    if g.has_edge(u, v):
        g.remove_edge(u, v)


def remove_node_if_exists(g: nx.Graph, v) -> None:
    if g.has_node(v):
        g.remove_node(v)


def read_movingai_map(path):
    # read a map given with the movingai-framework
    with open(path) as map_file:
        lines = map_file.readlines()
    height = int("".join([d for d in lines[1] if d in list("0123456789")]))
    width = int("".join([d for d in lines[2] if d in list("0123456789")]))

    graph = nx.grid_2d_graph(height, width)
    for edge in graph.edges():
        graph.edges()[edge]['dist'] = 1
    graph.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(width - 1) for y in range(height - 1)],
        dist=np.sqrt(2)
    )
    graph.add_edges_from(
        [((x + 1, y), (x, y + 1)) for x in range(width - 1) for y in range(height - 1)],
        dist=np.sqrt(2)
    )
    data = lines[4:]
    blocked = list("@OTW")
    for i, row in enumerate(data):
        for j, pixel in enumerate(row[:-1]):
            if pixel in blocked:
                remove_node_if_exists(graph, (i, j))
                remove_edge_if_exists(graph, (i + 1, j), (i, j + 1))
                remove_edge_if_exists(graph, (i - 1, j), (i, j + 1))
                remove_edge_if_exists(graph, (i + 1, j), (i, j - 1))
                remove_edge_if_exists(graph, (i - 1, j), (i, j - 1))

    for node in graph.nodes():
        graph.nodes()[node]["pos"] = node
    return graph, width, height, data


def gen_example_graph(a, b):
    if b > a:
        a, b = b, a
    example = nx.Graph()
    for i in range(a):
        example.add_node(chr(i + ord('a')), pos=(i, 0))
    for i in range(a - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    for i in range(a, a + b):
        example.add_node(chr(i + ord('a')), pos=(i - a + (a - b) / 2, 1))
    for i in range(a, a + b - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    example.add_edge(chr(ord('a') + int((a - b) / 2)), chr(ord('a') + a))
    example.add_edge(chr(ord('a') - int((a - b) / 2) - 1 + a), chr(ord('a') + a + b - 1))

    for source, sink in example.edges():
        p1 = np.array(example.nodes()[source]['pos'])
        p2 = np.array(example.nodes()[sink]['pos'])
        example.edges()[source, sink]['dist'] = np.linalg.norm(p1 - p2)

    return example


class Environment():
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple, planning_problem_parameters=PlanningProblemParameters()) -> None:
        self.planning_horizon = len(graph.nodes())
        self.planning_problem_parameters = planning_problem_parameters
        self.g = graph
        self.state = start
        self.start = start
        self.goal = goal
        self.perturbate_weights()
        for state in self.start + self.goal:
            assert(state in self.g.nodes())

    def perturbate_weights(self):
        for u, v in self.g.edges():
            if 'dist' not in self.g.edges[u, v]:
                self.g.edges[u, v]['dist'] = 1.0
            self.g.edges[u, v]['dist'] *= 1 + 0.00001 * np.random.randn()

    def get_graph(self) -> nx.Graph:
        return self.g
    
    def get_state_goal_tuples(self) -> list[tuple[int, int]]:
        """ Compute the start and goal pairs for each agent in the environment.

        Returns:
            tuple[int, int]: (start, goal)"""
        return [(s, g) for s, g in zip(self.state, self.goal) if s is not None]
    
    def __str__(self):
        return f"""Environment:
{self.g.number_of_nodes()} nodes, {self.g.number_of_edges()} edges
Start: {self.start}
Goal: {self.goal}
"""

    def get_position(self, state):
        if state is None:
            return
        if 'pos' in self.g.nodes()[state]:
            return self.g.nodes()[state]['pos']
    
    def get_next_goal(self, goal_index):
        return self.goal[goal_index]


class GraphEnvironment(Environment):
    def __init__(self, graph: nx.Graph, start: tuple, goal: tuple, planning_problem_parameters=PlanningProblemParameters()) -> None:
        super().__init__(graph, start, goal, planning_problem_parameters=planning_problem_parameters)


class MapfInfoEnvironment(Environment):
    def __init__(self, scenario_file, n_agents=None, planning_problem_parameters=PlanningProblemParameters()) -> None:
        graph, start, goal = None, None, None
        df = pd.read_csv(Path("benchmark") / "scen" / scenario_file,
                         sep="\t",
                         names=["id", "map_name", "w", "h", "x0", "y0", "x1", "y1", "cost"],
                         skiprows=1)
        self.width = df.w[0]
        self.height = df.h[0]
        self.map_file = Path() / "benchmark" / "maps" / df.map_name[0]
        self.scenario_file = scenario_file
        graph, w, h, data = read_movingai_map(self.map_file)

        sg = df.loc[:,"x0":"y1"].to_records(index=False)
        if n_agents is None:
            n_agents = len(sg)
        start = tuple((x, y) for y, x, *_ in sg[:n_agents])
        goal = tuple((x, y) for *_, y, x, in sg[:n_agents])
        for s in start:
            if s not in graph:
                logging.error(f"start {s} not in map {self.map_file}, value is {data[s[0]][s[1]]}")

        for g in goal:
            if g not in graph:
                logging.error(f"goal {g} not in map {self.map_file}, value is {data[s[0]][s[1]]}")
        super().__init__(graph, start, goal, planning_problem_parameters=planning_problem_parameters)

    def get_background_matrix(self):
        positions = nx.get_node_attributes(self.g, 'pos').values()
        positions = np.array(list(positions))
        image = np.zeros((self.width, self.height))
        for x, y in nx.get_node_attributes(self.g, 'pos').values():
            image[x, y] = 1
        return image

    def get_obstacle_df(self):
        m = self.get_background_matrix()
        data = []
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                if m[x, y] == 0:
                    data.append({'x': x, 'y': y, 'status': 'free'})
                else:
                    data.append({'x': x, 'y': y, 'status': 'occupied'})
        df = pd.DataFrame(data)
        df['status'] = df['status'].astype('category')
        return df


class RoadmapEnvironment(Environment):
    def __init__(self, 
                 map_path,
                 start_positions:tuple[float]|None,
                 goal_positions:tuple[float]|None,
                 n_agents:int|None=None,
                 generator_points=None, 
                 wx:tuple[float,float]|None=None, 
                 wy:tuple[float,float]|None=None, 
                 offset:float=0.15, 
                 planning_problem_parameters=PlanningProblemParameters()) -> None:
        """create an environment from robotics map with roadmap (voronoi based)

        :param map_path: path to map yaml
        :param start_positions: node ids! of start positions
        :param goal_positions: node ids! of goal positions
        :param generator_points: points that distribute space in voronoi cells, defaults to None
        :param wx: workspace extension in x, defaults to None
        :param wy: workspace extension in y, defaults to None
        :param offset: robot size (radius) + security, defaults to 0.15
        """
        _, obstacles = geometry.read_obstacles(map_path)
        if wx is None:
            wx = (-1, 3.5)
        if wy is None:
            wy = (-3, 1.5)
        if generator_points is None:
            generator_points = geometry.square_tiling(1.0, working_area_x=wx, working_area_y=wy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = geometry.create_graph(generator_points,
                                          working_area_x=wx,
                                          working_area_y=wy,
                                          offset=offset,
                                          occupied_space=obstacles)
        start_nodes = ()
        goal_nodes = ()
        if start_positions is not None:
            start_positions = start_positions[:n_agents]
        if goal_positions is not None:
            goal_positions = goal_positions[:n_agents]
        if start_positions is not None:
            start_nodes = tuple([geometry.find_nearest_node(graph, (x, y)) for x, y, *_ in start_positions])
            self._start_positions = start_positions
        if goal_positions is not None:
            goal_nodes = tuple([geometry.find_nearest_node(graph, (x, y)) for x, y, *_ in goal_positions])
            self._goal_positions = goal_positions

        self._obstacles = obstacles
        
        super().__init__(graph, start_nodes, goal_nodes, planning_problem_parameters=planning_problem_parameters)
        
    def plot(self, show=True, paths=None):
        import matplotlib.pyplot as plt
        if show:
            plt.figure()
        inner = self.get_inner_areas_gpd()
        inner.plot(ax=plt.gca(), alpha=0.5)
        connection = self.get_connection_gdp()
        connection.plot(ax=plt.gca())
        obstacles = self.get_obstacles_gpd()
        obstacles.plot(ax=plt.gca(), color='gray')
        start = gpd.GeoSeries([geometry.Point(x, y) for x, y, *_ in self._start_positions])
        start.plot(ax=plt.gca(), color='green')
        goal = gpd.GeoSeries([geometry.Point(x, y) for x, y, *_ in self._goal_positions])
        goal.plot(ax=plt.gca(), color='red')

        if paths is not None:
            for path in paths:
                path = gpd.GeoSeries([geometry.LineString([self.g.nodes()[n]['geometry'].center for n in path if n is not None])])
                path.plot(ax=plt.gca(), color='blue')
        if show:
            plt.show()
            plt.close()

    def get_obstacles_gpd(self):
        obstacles = gpd.GeoSeries(self._obstacles)
        return obstacles

    def get_connection_gdp(self):
        connection = gpd.GeoSeries(e['geometry'].connection for _, _, e in self.g.edges(data=True))
        return connection

    def get_inner_areas_gpd(self):
        inner = gpd.GeoSeries(n['geometry'].inner for _, n in self.g.nodes(data=True))
        return inner
    
    def find_nearest_node(self, pos):
        return geometry.find_nearest_node(self.g, pos)
        
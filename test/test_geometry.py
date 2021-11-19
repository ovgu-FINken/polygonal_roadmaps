import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
import numpy as np
from pathlib import Path

from polygonal_roadmaps import geometry


class TestGraphCreation(unittest.TestCase):
    def setUp(self):
        self.map_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "icra2021_map.yaml"

    def testReadMap(self):
        map, info = geometry.read_map(self.map_path)
        self.assertEqual(info['resolution'], 0.05)
        self.assertTrue(len(map))
        free, obstacles = geometry.read_obstacles(self.map_path)
        self.assertEqual(obstacles.geometryType(), 'MultiPolygon')
        self.assertEqual(free.geometryType(), 'MultiPolygon')
        self.assertFalse(free.buffer(-0.1).intersects(obstacles))

    def testGenGraph(self):
        _, obstacles = geometry.read_obstacles(self.map_path)
        wx = (-1, 3.5)
        wy = (-3, 1.5)
        generators = geometry.square_tiling(0.5, working_area_x=wx, working_area_y=wy)
        graph = geometry.create_graph(generators, working_area_x=wx, working_area_y=wy, occupied_space=obstacles, offset=0.15)
        self.assertGreater(graph.number_of_nodes(), 10)
        self.assertGreater(graph.number_of_edges(), 10)
        self.assertEqual(nx.number_connected_components(graph), 4)


class TestPathPolygon(unittest.TestCase):
    def setUp(self):
        map_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "icra2021_map.yaml"
        _, obstacles = geometry.read_obstacles(map_path)
        wx = (-1, 3.5)
        wy = (-3, 1.5)
        generators = geometry.square_tiling(0.5, working_area_x=wx, working_area_y=wy)
        self.graph = geometry.create_graph(generators, working_area_x=wx, working_area_y=wy, occupied_space=obstacles, offset=0.15)

    def testFindNearestNode(self):
        p1 = self.graph.nodes()[30]['geometry'].get_center_np()
        n1 = geometry.find_nearest_node(self.graph, p1)
        self.assertEqual(n1, 30)

        # see if we find the point despite a slight perturbation
        p2 = self.graph.nodes()[70]['geometry'].get_center_np()
        p2 += np.array([0.1, -0.1])
        n2 = geometry.find_nearest_node(self.graph, p2)
        self.assertEqual(n2, 70)

    def testPathFromPositions(self):
        start = .2, .2
        goal = .7, -1.7
        path = geometry.path_from_positions(self.graph, start, goal)
        self.assertEqual(path, [27, 26, 28, 30, 33, 64])
        # test non valid path (start and goal are not connected)
        goal = 1, 1
        self.assertRaises(nx.NetworkXNoPath, geometry.path_from_positions, self.graph, start, goal)

    def testPolyFromPath(self):
        path = [27, 26, 28, 30, 33, 64]
        poly = geometry.poly_from_path(self.graph, path, eps=0.05)
        self.assertTrue(poly.is_valid)
        self.assertEqual(poly.geometryType(), "Polygon")

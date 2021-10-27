import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
import numpy as np
from pathlib import Path
from polygonal_roadmaps import pathfinding
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
        free, obstacles = geometry.read_obstacles(self.map_path)
        wx = (-1, 3)
        wy = (-1, 3)
        generators = geometry.square_tiling(0.5, working_area_x=wx, working_area_y=wy)
        graph = geometry.create_graph(generators,working_area_x=wx, working_area_y=wy)
        self.assertTrue(True)
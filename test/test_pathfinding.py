import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from graph_tool.all import load_graph
from pathlib import Path
import os


from polygonal_roadmaps import pathfinding

class TestCBS(unittest.TestCase):
    def setUp(self):
        graph_file = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "test_graph.xml"
        self.graph = load_graph(str(graph_file))

    def testSumOfCost(self):
        result = pathfinding.sum_of_cost([[2], [2,3]])
        self.assertEqual(result, 3)

    def testPadPath(self):
        path = [1,2,3]
        path = pathfinding.pad_path(path, limit=100)
        self.assertEqual(len(path), 100)

    def testAStar(self):
        path = pathfinding.find_path_astar(self.graph, 0, 5)
        expected = [0, 1, 4, 5]
        self.assertEqual(path, expected)

        exception_thrown = False
        try:
            path = pathfinding.find_path_astar(self.graph, 0, 6)
        except pathfinding.PathDoesNotExistException:
            exception_thrown = True
        self.assertEqual(exception_thrown, True, "find_path_astar should raise PathDoesNotExistException, if the path is not found")

    def testSpaceTimeAStare(self):
        path = pathfinding.find_constrained_path(self.graph, 0, 5)
        expected = [0, 1, 4, 5]
        self.assertEqual(path, expected)

        path = pathfinding.find_constrained_path(self.graph, 0, 5, node_constraints=[pathfinding.NodeConstraint(agent=0, time=1, node=1)])
        expected = [0, 0, 1, 4, 5]
        self.assertEqual(path, expected)


        exception_thrown = False
        try:
            path = pathfinding.find_constrained_path(self.graph, 0, 6, node_constraints=[pathfinding.NodeConstraint(agent=0, time=1, node=1)])
        except pathfinding.PathDoesNotExistException:
            exception_thrown = True
        self.assertEqual(exception_thrown, True, "find_path_astar should raise PathDoesNotExistException, if the path is not found")
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from graph_tool import load_graph
from pathlib import Path
import os
import numpy as np


from polygonal_roadmaps import pathfinding

class TestLowLevelSearch(unittest.TestCase):
    def setUp(self):
        graph_file = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "test_graph.xml"
        self.graph = load_graph(str(graph_file))

    def testSumOfCost(self):
        result = pathfinding.sum_of_cost([[2], [2,3]])
        self.assertEqual(result, 3)
        
        result = pathfinding.sum_of_cost([[], [2,3]])
        self.assertEqual(result, np.inf)
        
        result = pathfinding.sum_of_cost(None)
        self.assertEqual(result, np.inf)
        
        result = pathfinding.sum_of_cost([[2,3], None])
        self.assertEqual(result, np.inf)

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

    def testSpaceTimeAStar(self):
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

class TestCBS(unittest.TestCase):
    def setUp(self):
        graph_file = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "test_graph.xml"
        self.graph = load_graph(str(graph_file))
        
    def test_compute_node_conflicts(self):
        path1 = [1,2]
        path2 = [2,1]
        conflicts = pathfinding.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(frozenset(), conflicts)
        
        path1 = [1,3,2]
        path2 = [2,3]
        conflicts = pathfinding.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(conflicts, frozenset([frozenset([pathfinding.NodeConstraint(agent=0,time=1,node=3), pathfinding.NodeConstraint(agent=1,time=1,node=3)])]))
        
        conflicts = pathfinding.compute_node_conflicts([[0,1,4,5],[2,3]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

        conflicts = pathfinding.compute_node_conflicts([[0,1,4,5],[5,4,1,0]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

    def test_compute_edge_conflicts(self):
        path1 = [1,2]
        path2 = [3,4]
        conflicts = pathfinding.compute_edge_conflicts([path1, path2], limit=10)
        self.assertEqual(conflicts, frozenset())

        
        conflicts = pathfinding.compute_edge_conflicts([[0,1,4,5],[2,3]], limit=10)
        self.assertEqual(conflicts, frozenset())



        #following conflict
        path1 = [1,2,3]
        path2 = [2,3,4]
        conflicts = pathfinding.compute_edge_conflicts([path1, path2], limit=10)
        self.assertEqual(
            conflicts, 
            frozenset([ 
                frozenset([pathfinding.NodeConstraint(agent=0, time=1, node=2)]),
                frozenset([pathfinding.NodeConstraint(agent=0, time=2, node=3), pathfinding.NodeConstraint(agent=1, time=1, node=3)])
            ])
        )

        #swapping conflict
        path1 = [1,2,3,4]
        path2 = [4,3,2,1]
        conflicts = pathfinding.compute_edge_conflicts([path1, path2], limit=10)
        expected = frozenset([
                frozenset([pathfinding.NodeConstraint(agent=0, time=2, node=3), pathfinding.NodeConstraint(agent=1, time=1, node=3)]),
                frozenset([pathfinding.NodeConstraint(agent=1, time=2, node=2), pathfinding.NodeConstraint(agent=0, time=1, node=2)])
            ])
        self.assertEqual(conflicts, expected)


    def testNonValidPath(self):
        cbs = pathfinding.CBS(self.graph, [(0,6)])
        self.assertRaises(pathfinding.PathDoesNotExistException, cbs.run)
    
    def testCBSstep(self):
        cbs = pathfinding.CBS(self.graph, [(0,5), (5,0)], limit=100)
        # the first step should expand the root node of the constraint tree
        self.assertTrue(cbs.root.open)
        cbs.step()
        self.assertFalse(cbs.root.open)
        self.assertLess(cbs.root.fitness, 2 * 100) # fitness should be less than agents * limit
        # our example has conflicts, so the root node should not be the final node
        self.assertEqual(cbs.root.solution, [[0,1,4,5], [5,4,1,0]])
        expected = frozenset([
            frozenset([pathfinding.NodeConstraint(agent=0, time=2, node=4), pathfinding.NodeConstraint(agent=1, time=1, node=4)]),
            frozenset([pathfinding.NodeConstraint(agent=1, time=2, node=1), pathfinding.NodeConstraint(agent=0, time=1, node=1)])
        ])
        self.assertEqual(cbs.root.conflicts, expected)
        
        self.assertFalse(cbs.root.final)
        
    def testTraversal(self):
        node1 = pathfinding.CBSNode()
        node1.tag = 1
        node2 = pathfinding.CBSNode()
        node2.tag = 2
        node3 = pathfinding.CBSNode()
        node3.tag = 3
        node4 = pathfinding.CBSNode()
        node4.tag = 4
        node5 = pathfinding.CBSNode()
        node5.tag = 5
        
        node1.children = (node2, node3)
        node2.children = (node4)
        node3.children = (node5)
        
        traversal = [x.tag for x in node1]
        self.assertEqual(traversal, [1,2,4,3,5])

    def testCBSNoConflict(self):
        cbs = pathfinding.CBS(self.graph, [(0,5), (2,3)], limit=100)
        try:
            best = cbs.run()
        except pathfinding.PathDoesNotExistException:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(cbs.root.solution, [[0,1,4,5],[2,3]])
        self.assertEqual(cbs.root.conflicts, frozenset())
        self.assertTrue(cbs.root.final, "The root node should be the final node, as there are no conflicts")


    def testCBSrun(self):
        cbs = pathfinding.CBS(self.graph, [(0,5), (5,0)], limit=20)
        exception_raised = False
        try:
            best = cbs.run()
        except pathfinding.PathDoesNotExistException:
            exception_raised = True

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")

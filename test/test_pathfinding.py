import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import networkx as nx
from pathlib import Path
import os
import numpy as np


from polygonal_roadmaps import pathfinding

def load_graph(filename):
    return nx.read_graphml(filename)

class TestLowLevelSearch(unittest.TestCase):
    def setUp(self):
        self.graph = pathfinding.gen_example_graph(5,2)
        

    def testSumOfCost(self):
        result = pathfinding.sum_of_cost([['b'], ['b', 'c']])
        self.assertEqual(result, 3)
        
        result = pathfinding.sum_of_cost([[], ['2','3']])
        self.assertEqual(result, np.inf)
        
        result = pathfinding.sum_of_cost(None)
        self.assertEqual(result, np.inf)
        
        result = pathfinding.sum_of_cost([[2,3], None])
        self.assertEqual(result, np.inf)
        
        result = pathfinding.sum_of_cost(['abcde','cc'], graph=self.graph)
        self.assertAlmostEqual(result, 4.1, places=1)

    def testPadPath(self):
        path = [1,2,3]
        path = pathfinding.pad_path(path, limit=100)
        self.assertEqual(len(path), 100)

    def testAStar(self):
        path = pathfinding.spatial_astar(self.graph, 'a', 'e')
        expected = list('abcde')
        self.assertEqual(path, expected)
        
        path = pathfinding.spatial_astar(self.graph, 'c', 'a')
        expected = list('cba')
        self.assertEqual(path, expected)


        self.assertRaises(nx.NodeNotFound, pathfinding.spatial_astar, self.graph, 'a', 'Z')
        self.graph.add_node('Z', pos=(0.5,0.5))
        self.assertRaises(nx.NetworkXNoPath, pathfinding.spatial_astar, self.graph, 'a', 'Z')


    def testSpaceTimeAStar(self):
        path = pathfinding.spacetime_astar(self.graph, 'a', 'e', None)
        expected = list('abcde')
        self.assertEqual(path, expected)
        
        path = pathfinding.spacetime_astar(self.graph, 'c', 'a', None)
        expected = list('cba')
        self.assertEqual(path, expected)


        self.assertRaises(nx.NodeNotFound, pathfinding.spacetime_astar, self.graph, 'a', 'Z', None)
        self.graph.add_node('Z', pos=(0.5,0.5))
        self.assertRaises(nx.NetworkXNoPath, pathfinding.spacetime_astar, self.graph, 'a', 'Z', None), None
        

        cost = pathfinding.compute_cost(self.graph, 'e')
        path = pathfinding.spacetime_astar(self.graph, 'a', 'e', cost, node_constraints=[('b', 1)])
        expected = list('aabcde')
        self.assertEqual(path, expected)
        
        path = pathfinding.spacetime_astar(self.graph, 'a', 'e', cost, node_constraints=[('c', 2)])
        expected = [ list('aabcde'), list('abfgde')]
        self.assertIn(path, expected)
        
        cost = pathfinding.compute_cost(self.graph, 'b')
        path = pathfinding.spacetime_astar(self.graph, 'd', 'b', cost, node_constraints=[('b', 3)])
        expected = list('dcb')
        self.assertEqual(path, expected)


        

class TestCBS(unittest.TestCase):
    def setUp(self):
        self.graph = pathfinding.gen_example_graph(5,2)
        
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
        # when the node is not within the graph, an exception is raised, because cost computation fails
        self.assertRaises(nx.NodeNotFound, pathfinding.CBS, self.graph, [('a','Z')])

    
    def testCBSsetup(self):
        cbs = pathfinding.CBS(self.graph, [('a','e'), ('e','a')], limit=15)
        cbs.setup()
        # the first step should expand the root node of the constraint tree
        self.assertFalse(cbs.root.open)
        self.assertLess(cbs.root.fitness, 2 * 100) # fitness should be less than agents * limit
        # our example has conflicts, so the root node should not be the final node
        self.assertEqual(cbs.root.solution, [(list("abcde")), list("edcba")])
        expected = frozenset([
            pathfinding.NodeConstraint(agent=0, time=2, node='c'), pathfinding.NodeConstraint(agent=1, time=2, node='c')
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
        cbs = pathfinding.CBS(self.graph, [('b', 'e'), ('g','a')], limit=15)
        try:
            best = cbs.run()
        except pathfinding.PathDoesNotExistException:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(cbs.root.solution, [list('bcde'),list('gfba')])
        self.assertEqual(cbs.root.conflicts, frozenset())
        self.assertTrue(cbs.root.final, "The root node should be the final node, as there are no conflicts")


    def testCBSrun(self):
        cbs = pathfinding.CBS(self.graph, [('a', 'e'), ('e', 'a')], limit=12)
        exception_raised = False
        try:
            best = cbs.run()
        except nx.NetworkXNoPath:
            exception_raised = True
        

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")
        self.assertIn(best, ( [['a', 'b', 'f', 'g', 'd', 'e'], ['e', 'd', 'c', 'b', 'a']], [['a', 'b', 'c', 'd', 'e'], ['e', 'd', 'g', 'f', 'b', 'a']]) )

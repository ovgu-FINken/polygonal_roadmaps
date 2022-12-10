import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import networkx as nx
import os
import numpy as np


from polygonal_roadmaps import planner
from polygonal_roadmaps.environment import gen_example_graph


def load_graph(filename):
    return nx.read_graphml(filename)


class TestLowLevelSearch(unittest.TestCase):
    def setUp(self):
        self.graph = gen_example_graph(5, 2)

    def testSumOfCost(self):
        result = planner.sum_of_cost([['b'], ['b', 'c']])
        self.assertEqual(result, 3)

        # this is the behaviour of the algorithm, when one agent is at its goal already
        result = planner.sum_of_cost([[], ['2', '3']])
        self.assertEqual(result, 2)

        result = planner.sum_of_cost(None)
        self.assertEqual(result, np.inf)

        result = planner.sum_of_cost([[2, 3], None])
        self.assertEqual(result, np.inf)

        result = planner.sum_of_cost(['abcde', 'cc'], graph=self.graph, weight="dist")
        self.assertAlmostEqual(result, 5.0, places=1)

    def testPadPath(self):
        path = [1, 2, 3]
        path = planner.pad_path(path, limit=100)
        self.assertEqual(len(path), 100)

    def testAStar(self):
        path = planner.spatial_astar(self.graph, 'a', 'e')
        expected = list('abcde')
        self.assertEqual(path, expected)

        path = planner.spatial_astar(self.graph, 'c', 'a')
        expected = list('cba')
        self.assertEqual(path, expected)

        self.assertRaises(nx.NodeNotFound, planner.spatial_astar, self.graph, 'a', 'Z')
        self.graph.add_node('Z', pos=(0.5, 0.5))
        self.assertRaises(nx.NetworkXNoPath, planner.spatial_astar, self.graph, 'a', 'Z')

    def testSpaceTimeAStar(self):
        for n1, n2 in self.graph.edges():
            self.graph.edges()[n1, n2]['weight'] = self.graph.edges()[n1, n2]['dist']
        path, _ = planner.spacetime_astar(self.graph, 'a', 'e', None)
        expected = list('abcde')
        self.assertEqual(path, expected)

        path, _ = planner.spacetime_astar(self.graph, 'c', 'a', None)
        expected = list('cba')
        self.assertEqual(path, expected)

        self.assertRaises(nx.NodeNotFound, planner.spacetime_astar, self.graph, 'a', 'Z', None)
        self.graph.add_node('Z', pos=(0.5, 0.5))
        self.assertRaises(nx.NetworkXNoPath, planner.spacetime_astar, self.graph, 'a', 'Z', None), None

        path, _ = planner.spacetime_astar(self.graph, 'a', 'e', node_constraints=[('b', 1)])
        expected = list('aabcde')
        self.assertEqual(path, expected)

        path, _ = planner.spacetime_astar(self.graph, 'a', 'e', node_constraints=[('c', 2)])
        expected = [list('aabcde'), list('abbcde'), list('abfgde')]
        self.assertIn(path, expected)

        path, _ = planner.spacetime_astar(self.graph, 'd', 'b', node_constraints=[('b', 3)])
        expected = list('dcb')
        self.assertEqual(path, expected)


class TestPrioritizedSearch(unittest.TestCase):
    def setUp(self):
        self.graph = gen_example_graph(5, 2)

    def testNoConflict(self):
        try:
            solution = planner.prioritized_plans(self.graph, [('b', 'e'), ('g', 'a')], limit=15)
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(solution, [list('bcde'), list('gfba')])

    def testConflict(self):
        try:
            solution = planner.prioritized_plans(self.graph, [('a', 'e'), ('e', 'a')], limit=15)
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(solution, [list('abcde'), list('edgfba')])


class TestCBS(unittest.TestCase):
    def setUp(self):
        self.graph = gen_example_graph(5, 2)

    def test_compute_node_conflicts(self):
        path1 = [1, 2]
        path2 = [2, 1]
        conflicts = planner.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(frozenset(), conflicts)

        path1 = [1, 3, 2]
        path2 = [2, 3]
        conflicts = planner.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(conflicts, frozenset(
            [frozenset([planner.NodeConstraint(agent=0, time=1, node=3),
             planner.NodeConstraint(agent=1, time=1, node=3)])]))

        conflicts = planner.compute_node_conflicts([[0, 1, 4, 5], [2, 3]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

        conflicts = planner.compute_node_conflicts([[0, 1, 4, 5], [5, 4, 1, 0]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

    def test_compute_1_robust_conflicts(self):
        path1 = [1, 2]
        path2 = [3, 4]
        conflicts = planner.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        self.assertEqual(conflicts, frozenset())

        conflicts = planner.compute_k_robustness_conflicts([[0, 1, 4, 5], [2, 3]], limit=10, k=1)
        self.assertEqual(conflicts, frozenset())

        # following conflict
        path1 = [1, 2, 3]
        path2 = [2, 3, 4]
        conflicts = planner.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        self.assertEqual(
            conflicts,
            frozenset([
                planner.Conflict(k=1, conflicting_agents=frozenset([planner.NodeConstraint(agent=0, time=1, node=2)])),
                planner.Conflict(k=1, conflicting_agents=frozenset(
                    [planner.NodeConstraint(agent=0, time=2, node=3),
                     planner.NodeConstraint(agent=1, time=1, node=3)]))
            ])
        )

        # swapping conflict
        path1 = [1, 2, 3, 4]
        path2 = [4, 3, 2, 1]
        conflicts = planner.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        expected = frozenset([
            planner.Conflict(k=1, conflicting_agents=frozenset(
                [planner.NodeConstraint(agent=0, time=2, node=3),
                 planner.NodeConstraint(agent=1, time=1, node=3)])),
            planner.Conflict(k=1, conflicting_agents=frozenset(
                [planner.NodeConstraint(agent=1, time=2, node=2),
                 planner.NodeConstraint(agent=0, time=1, node=2)]))
        ])
        self.assertEqual(conflicts, expected)

    def testNonValidPath(self):
        # when the node is not within the graph, an exception is raised, because cost computation fails
        self.assertRaises(nx.NodeNotFound, planner.CBS, self.graph, [('a', 'Z')])
        self.graph.add_node('X', pos=(0.1, 0.1))
        cbs = planner.CBS(self.graph, [('a', 'b'), ('a', 'X')])
        self.assertRaises(nx.NetworkXNoPath, cbs.run)

    def testCBSsetup(self):
        cbs = planner.CBS(self.graph, [('a', 'e'), ('e', 'a')], limit=15)
        cbs.setup()
        # the first step should expand the root node of the constraint tree
        self.assertTrue(cbs.root.open)
        self.assertLess(cbs.root.fitness, 2 * 100)  # fitness should be less than agents * limit
        # our example has conflicts, so the root node should not be the final node
        self.assertEqual(cbs.root.solution, [(list("abcde")), list("edcba")])
        self.assertFalse(cbs.root.final)

    def testTraversal(self):
        node1 = planner.CBSNode()
        node1.tag = 1
        node2 = planner.CBSNode()
        node2.tag = 2
        node3 = planner.CBSNode()
        node3.tag = 3
        node4 = planner.CBSNode()
        node4.tag = 4
        node5 = planner.CBSNode()
        node5.tag = 5

        node1.children = (node2, node3)
        node2.children = (node4)
        node3.children = (node5)

        traversal = [x.tag for x in node1]
        self.assertEqual(traversal, [1, 2, 4, 3, 5])

    def testCBSNoConflict(self):
        cbs = planner.CBS(self.graph, [('b', 'e'), ('g', 'a')], limit=15)
        try:
            cbs.run()
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(cbs.root.solution, [list('bcde'), list('gfba')])
        self.assertEqual(cbs.root.conflicts, frozenset())
        self.assertTrue(cbs.root.final, "The root node should be the final node, as there are no conflicts")

    def testCBSrun(self):
        cbs = planner.CBS(self.graph, [('a', 'e'), ('e', 'a')], limit=12)
        exception_raised = False
        try:
            best = cbs.run()
        except nx.NetworkXNoPath:
            exception_raised = True

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")
        self.assertIn(best.solution, ([['a', 'b', 'f', 'g', 'd', 'e'], ['e', 'd', 'c', 'b', 'a']],
                      [['a', 'b', 'c', 'd', 'e'], ['e', 'd', 'g', 'f', 'b', 'a']]))

        G = gen_example_graph(5, 3)
        cbs = planner.CBS(G, [('b', 'e'), ('e', 'a'), ('a', 'f')], limit=28)
        exception_raised = False
        try:
            cbs.run()
        except nx.NetworkXNoPath:
            exception_raised = True

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")


class TestCCR(unittest.TestCase):
    def testDirectComparison(self):
        qualities = [np.array([-1, -1]), np.array([-1, -1]), np.array([-0.5, -1]), np.array([1.0, 0])]
        decision = planner.decision_function(qualities, method='direct_comparison')
        self.assertEqual(decision, 3)

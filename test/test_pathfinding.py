import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import networkx as nx
import os
import numpy as np


from polygonal_roadmaps import planning
from polygonal_roadmaps.environment import PlanningProblemParameters, gen_example_graph, GraphEnvironment


def load_graph(filename):
    return nx.read_graphml(filename)


class TestLowLevelSearch(unittest.TestCase):
    def setUp(self):
        self.graph = gen_example_graph(5, 2)

    def testSumOfCost(self):
        result = planning.sum_of_cost([['b'], ['b', 'c']])
        self.assertEqual(result, 3)

        # this is the behaviour of the algorithm, when one agent is at its goal already
        result = planning.sum_of_cost([[], ['2', '3']])
        self.assertEqual(result, 2)

        result = planning.sum_of_cost(None)
        self.assertEqual(result, np.inf)

        result = planning.sum_of_cost([[2, 3], None])
        self.assertEqual(result, np.inf)

        result = planning.sum_of_cost(['abcde', 'cc'], graph=self.graph, weight="dist")
        self.assertAlmostEqual(result, 5.0, places=1)

    def testPadPath(self):
        path = [1, 2, 3]
        path = planning.pad_path(path, limit=100)
        self.assertEqual(len(path), 100)

    def testAStar(self):
        path = planning.spatial_astar(self.graph, 'a', 'e')
        expected = list('abcde')
        self.assertEqual(path, expected)

        path = planning.spatial_astar(self.graph, 'c', 'a')
        expected = list('cba')
        self.assertEqual(path, expected)

        self.assertRaises(nx.NodeNotFound, planning.spatial_astar, self.graph, 'a', 'Z')
        self.graph.add_node('Z', pos=(0.5, 0.5))
        self.assertRaises(nx.NetworkXNoPath, planning.spatial_astar, self.graph, 'a', 'Z')

    def testSpaceTimeAStar(self):
        for n1, n2 in self.graph.edges():
            self.graph.edges()[n1, n2]['weight'] = self.graph.edges()[n1, n2]['dist']
        path, _ = planning.spacetime_astar(self.graph, 'a', 'e', None)
        expected = list('abcde')
        self.assertEqual(path, expected)

        path, _ = planning.spacetime_astar(self.graph, 'c', 'a', None)
        expected = list('cba')
        self.assertEqual(path, expected)

        self.assertRaises(nx.NodeNotFound, planning.spacetime_astar, self.graph, 'a', 'Z', None)
        self.graph.add_node('Z', pos=(0.5, 0.5))
        self.assertRaises(nx.NetworkXNoPath, planning.spacetime_astar, self.graph, 'a', 'Z', None), None

        path, _ = planning.spacetime_astar(self.graph, 'a', 'e', node_constraints=[('b', 1)])
        expected = list('aabcde')
        self.assertEqual(path, expected)

        path, _ = planning.spacetime_astar(self.graph, 'a', 'e', node_constraints=[('c', 2)])
        expected = [list('aabcde'), list('abbcde'), list('abfgde')]
        self.assertIn(path, expected)

        path, _ = planning.spacetime_astar(self.graph, 'd', 'b', node_constraints=[('b', 3)])
        expected = list('dcb')
        self.assertEqual(path, expected)


class TestPrioritizedSearch(unittest.TestCase):
    def setUp(self):
        self.env = GraphEnvironment(graph=gen_example_graph(5, 2), start=('b', 'g'), goal=('e', 'a'), planning_problem_parameters=PlanningProblemParameters(max_distance=12))

    def testNoConflict(self):
        try:
            solution = planning.prioritized_plans(self.env)
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(solution, [list('bcde'), list('gfba')])

    def testConflict(self):
        try:
            self.env.state = ('a', 'e')
            solution = planning.prioritized_plans(self.env)
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(solution, [list('abcde'), list('edgfba')])


class TestCBS(unittest.TestCase):
    def setUp(self):
        self.environment = GraphEnvironment(graph=gen_example_graph(5, 2), start=('a', 'b'), goal=('b', 'a'))

    def test_compute_node_conflicts(self):
        path1 = [1, 2]
        path2 = [2, 1]
        conflicts = planning.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(frozenset(), conflicts)

        path1 = [1, 3, 2]
        path2 = [2, 3]
        conflicts = planning.compute_node_conflicts([path1, path2], limit=10)
        self.assertEqual(conflicts, frozenset(
            [frozenset([planning.NodeConstraint(agent=0, time=1, node=3),
             planning.NodeConstraint(agent=1, time=1, node=3)])]))

        conflicts = planning.compute_node_conflicts([[0, 1, 4, 5], [2, 3]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

        conflicts = planning.compute_node_conflicts([[0, 1, 4, 5], [5, 4, 1, 0]], limit=10)
        self.assertEqual(conflicts, frozenset([]))

    def test_compute_1_robust_conflicts(self):
        path1 = [1, 2]
        path2 = [3, 4]
        conflicts = planning.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        self.assertEqual(conflicts, frozenset())

        conflicts = planning.compute_k_robustness_conflicts([[0, 1, 4, 5], [2, 3]], limit=10, k=1)
        self.assertEqual(conflicts, frozenset())

        # following conflict
        path1 = [1, 2, 3]
        path2 = [2, 3, 4]
        conflicts = planning.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        self.assertEqual(
            conflicts,
            frozenset([
                planning.Conflict(k=1, conflicting_agents=frozenset([planning.NodeConstraint(agent=0, time=1, node=2)])),
                planning.Conflict(k=1, conflicting_agents=frozenset(
                    [planning.NodeConstraint(agent=0, time=2, node=3),
                     planning.NodeConstraint(agent=1, time=1, node=3)]))
            ])
        )

        # swapping conflict
        path1 = [1, 2, 3, 4]
        path2 = [4, 3, 2, 1]
        conflicts = planning.compute_k_robustness_conflicts([path1, path2], limit=10, k=1)
        expected = frozenset([
            planning.Conflict(k=1, conflicting_agents=frozenset(
                [planning.NodeConstraint(agent=0, time=2, node=3),
                 planning.NodeConstraint(agent=1, time=1, node=3)])),
            planning.Conflict(k=1, conflicting_agents=frozenset(
                [planning.NodeConstraint(agent=1, time=2, node=2),
                 planning.NodeConstraint(agent=0, time=1, node=2)]))
        ])
        self.assertEqual(conflicts, expected)

    def testNonValidPath(self):
        # when the node is not within the graph, an exception is raised, because cost computation fails
        self.environment.goal = ('a', 'Z')
        self.assertRaises(nx.NodeNotFound, planning.CBS, self.environment)
        self.environment.g.add_node('X', pos=(0.1, 0.1))
        self.environment.goal = ('a', 'X')
        cbs = planning.CBS(self.environment)
        self.assertRaises(nx.NetworkXNoPath, cbs.run)

    def testCBSsetup(self):
        self.environment.goal = 'e', 'a'
        self.environment.start = 'a', 'e'
        self.environment.state = 'a', 'e'
        cbs = planning.CBS(self.environment)
        cbs.setup()
        # the first step should expand the root node of the constraint tree
        self.assertTrue(cbs.root.open)
        self.assertLess(cbs.root.fitness, 2 * 100)  # fitness should be less than agents * limit
        # our example has conflicts, so the root node should not be the final node
        self.assertEqual(cbs.root.solution, [(list("abcde")), list("edcba")])
        self.assertFalse(cbs.root.final)

    def testTraversal(self):
        node1 = planning.CBSNode()
        node1.tag = 1
        node2 = planning.CBSNode()
        node2.tag = 2
        node3 = planning.CBSNode()
        node3.tag = 3
        node4 = planning.CBSNode()
        node4.tag = 4
        node5 = planning.CBSNode()
        node5.tag = 5

        node1.children = (node2, node3)
        node2.children = (node4)
        node3.children = (node5)

        traversal = [x.tag for x in node1]
        self.assertEqual(traversal, [1, 2, 4, 3, 5])

    def testCBSNoConflict(self):
        self.environment.start = 'b', 'g'
        self.environment.state = 'b', 'g'
        self.environment.goal = 'e', 'a'
        cbs = planning.CBS(self.environment)
        try:
            cbs.run()
        except nx.NetworkXNoPath:
            self.assertTrue(False, msg="exception should not be raised, as path is valid")
        self.assertEqual(cbs.root.solution, [list('bcde'), list('gfba')])
        self.assertEqual(cbs.root.conflicts, frozenset())
        self.assertTrue(cbs.root.final, "The root node should be the final node, as there are no conflicts")

    def testCBSrun(self):
        self.environment.goal = 'e', 'a'
        self.environment.start = 'a', 'e'
        self.environment.state = 'a', 'e'
        cbs = planning.CBS(self.environment)
        exception_raised = False
        try:
            best = cbs.run()
        except nx.NetworkXNoPath:
            exception_raised = True

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")
        self.assertIn(best.solution, ([['a', 'b', 'f', 'g', 'd', 'e'], ['e', 'd', 'c', 'b', 'a']],
                      [['a', 'b', 'c', 'd', 'e'], ['e', 'd', 'g', 'f', 'b', 'a']]))

        env = GraphEnvironment(gen_example_graph(5, 3), start=('b', 'e', 'a'), goal=('e', 'a', 'f'))
        cbs = planning.CBS(env, planning.PlanningProblemParameters(max_distance=28))
        exception_raised = False
        try:
            cbs.run()
        except nx.NetworkXNoPath:
            exception_raised = True

        self.assertFalse(exception_raised, msg="exception should not be raised, as path is valid")


class TestCCR(unittest.TestCase):
    def testDirectComparison(self):
        qualities = [np.array([-1, -1]), np.array([-1, -1]), np.array([-0.5, -1]), np.array([1.0, 0])]
        decision = planning.decision_function(qualities, method='direct_comparison')
        self.assertEqual(decision, 3)

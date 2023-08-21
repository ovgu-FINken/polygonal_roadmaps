import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import networkx as nx
import os
import numpy as np


from polygonal_roadmaps import planning, cli
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
        
    def testComputeAllKConflicts(self):
        p1 = [1, 2, 3, 4, 5]
        p2 = [5, 4, 3, 2, 1]
        conflicts = planning.compute_all_k_conflicts([p1, p2], limit=10, k=1)
        # there should be a k=0 conflict at node 3:
        expected = frozenset([ planning.Conflict(k=0, conflicting_agents=frozenset({
            planning.NodeConstraint(agent=1, time=2, node=3),
            planning.NodeConstraint(agent=0, time=2, node=3)}) )
        ])
        self.assertEqual(conflicts, expected)
        p1 = [1, 1, 1, 3, 1, 1]
        p2 = [2, 2, 3, 2, 2]
        conflicts = planning.compute_all_k_conflicts([p1, p2], limit=10, k=1)
        # there should be a k=1 conflict at node 3:
        expected = frozenset([ planning.Conflict(k=1, conflicting_agents=frozenset({
            planning.NodeConstraint(agent=1, time=2, node=3),
            planning.NodeConstraint(agent=0, time=3, node=3)}) )
        ])
        self.assertEqual(conflicts, expected)


class TestPrioritizedSearch(unittest.TestCase):
    def setUp(self):
        self.env = GraphEnvironment(graph=gen_example_graph(5, 2), start=('b', 'g'), goal=('e', 'a'))

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


class TestCCRv2(unittest.TestCase):
    def setUp(self):
        self.env = GraphEnvironment(graph=gen_example_graph(5, 2), start=('b', 'g'), goal=('e', 'a'))
        self.env2 = cli.env_generator('DrivingSwarm;icra2021_map.yaml;icra2021.yml', n_agents= 3)[0]

    def testNoConflict(self):
        planner = planning.CCRv2(self.env)
        plan = planner.create_plan()
        reference = list(zip(list('bcde') + [None], list('gfba') + [None]))
        self.assertEqual(plan, reference)

    def testConflict(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        plan = planner.create_plan()
        reference = list(zip(list('abcde') + [None], list('edcba')+ [None]))
        self.assertTrue(planner.agents[0].is_consistent())
        self.assertTrue(planner.agents[1].is_consistent())
        self.assertEqual(len(planner.agents[0].get_conflicts()), 0)
        self.assertEqual(len(planner.agents[1].get_conflicts()), 0)
        self.assertNotEqual(plan, reference)
        
        planner = planning.CCRv2(self.env2)
        plan = planner.create_plan()
        self.assertTrue(planner.agents[0].is_consistent())
        self.assertTrue(planner.agents[1].is_consistent())
        self.assertTrue(planner.agents[2].is_consistent())
        
    def testAgentInit(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        self.assertListEqual(planner.agents[0].plan, list('abcde'))
        self.assertTrue(planner.agents[0].is_consistent())
        self.assertFalse(planner.agents[0].get_conflicts())
        
    def testCheckConflicts(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        # check the assumption of the test
        self.assertListEqual(planner.agents[0].plan, list('abcde'))
        # create a conflict at node c
        planner.agents[0].update_other_paths({1: list('edcba')})
        self.assertGreater(len(planner.agents[0].conflicts), 0)
        # still the conflicts is consistent with the belief of the agent
        # the node has not been visited yet by CDM and there is no belief for this state
        self.assertTrue(planner.agents[0].is_consistent())

        # no we add the belief state for the node C
        # b-c has lower priority than c-d, thus for agent[0] this is not consistent
        bs = planning.BeliefState('c', {'b': 0.5, 'd': 1.5})
        planner.agents[0].set_belief('c', bs)
        self.assertFalse(planner.agents[0].is_consistent())

        # if we reverse the priorities, the agent is consistent again
        bs = planning.BeliefState('c', {'b': 1.5, 'd': 0.5})
        planner.agents[0].set_belief('c', bs)
        self.assertTrue(planner.agents[0].is_consistent())
                 
    def testUpdatePath(self):
        planner = planning.CCRv2(self.env)
        # should not add own path
        planner.agents[0].update_other_paths({0: list('abcde')})
        self.assertEqual(planner.agents[0].other_paths, {})
        # other plan ends up in other_paths
        planner.agents[0].plan = list('edcba')
        planner.agents[0].update_other_paths({1: list('edcba')})
        self.assertEqual(planner.agents[0].other_paths, {1: list('edcba')})
        # plans get overwritten when updated
        planner.agents[0].update_other_paths({1: list('abcde')})
        self.assertEqual(planner.agents[0].other_paths, {1: list('abcde')})

    def testCDMUpdate(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        # check the assumption of the test
        self.assertListEqual(planner.agents[0].plan, list('abcde'))
        self.assertListEqual(planner.agents[1].plan, list('edcba'))
        # intially plans are consistent, because there are conflicts, but no belief about priorities
        planner.update_all_paths()
        self.assertGreaterEqual(len(planner.agents[0].get_conflicts()), 0)
        self.assertGreaterEqual(len(planner.agents[1].get_conflicts()), 0)
        # making consistent does not lead to change
        self.assertFalse(planner.make_all_plans_consistent())
        # cdm is performed:
        node, _ = planner.make_cdm_decision()
        decsion = planning.BeliefState('c', {'b': 0.5, 'd': 1.5})
        for a in planner.agents:
            a.set_belief(node, decsion)
        # now the plan for a[0] is not consistent
        self.assertFalse(planner.agents[0].is_consistent())
        # plan for a[1] is still consistent, because he gets prio
        self.assertTrue(planner.agents[1].is_consistent())
        planner.make_all_plans_consistent()
        self.assertTrue(planner.agents[0].is_consistent())
        self.assertTrue(planner.agents[1].is_consistent())
        
    def testMakePlansConsistent(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        planner.update_all_paths()
        # cdm is performed:
        node, _ = planner.make_cdm_decision()
        decsion = planning.BeliefState('c', {'b': 0.5, 'd': 1.5})
        for a in planner.agents:
            a.set_belief(node, decsion)
        # check the assumption of the test
        # now the plan for a[0] is not consistent
        self.assertFalse(planner.agents[0].is_consistent())
        # plan for a[1] is still consistent, because he gets prio
        self.assertTrue(planner.agents[1].is_consistent())
        p0 = planner.agents[0].get_plan()
        changed = planner.agents[0].make_plan_consistent(recurse=True)
        self.assertTrue(changed)
        self.assertIn(planning.NodeConstraint(agent=0, time=2, node='c'), planner.agents[0].constraints)
        self.assertIn(planning.NodeConstraint(agent=0, time=1, node='c'), planner.agents[0].constraints)
        self.assertIn(planning.NodeConstraint(agent=0, time=3, node='c'), planner.agents[0].constraints)
        p1 = planner.agents[0].get_plan()
        self.assertNotEqual(p0, p1)
        self.assertEqual(p1, list('abfgde'))
        self.assertTrue(planner.agents[0].is_consistent())
        
        
    def testCDMOpinion(self):
        self.env.state = ('a', 'e')
        planner = planning.CCRv2(self.env)
        # check the assumption of the test
        # check the assumption of the test
        self.assertListEqual(planner.agents[0].plan, list('abcde'))
        self.assertListEqual(planner.agents[1].plan, list('edcba'))
        planner.update_all_paths()
        # conflcit at node c
        # check the opinion function:
        bs = planner.agents[0].get_cdm_opinion('c')
        self.assertEqual(bs.state, 'c')
        self.assertIn('c', bs.priorities)
        self.assertIn('b', bs.priorities)
        self.assertIn('d', bs.priorities)
        # cdm is performed:
        node, decision = planner.make_cdm_decision()
        self.assertEqual(node, 'c') # only node with a conflict
        self.assertTrue(type(decision) is planning.BeliefState)
        self.assertEqual(decision.state, 'c')
        self.assertIn('c', decision.priorities)        
        self.assertIn('b', decision.priorities)        
        self.assertIn('d', decision.priorities)        
    
    def testBeliefState(self):
        bs1 = planning.BeliefState('c', {'b': 0.5, 'd': 1.5})
        bs2 = planning.BeliefState('c', {'b': 1.5, 'd': 0.5})
        bs_plus = bs1 + bs2
        self.assertDictEqual(bs_plus.priorities, {'b': 2.0, 'd': 2.0, 'c': np.inf})
        bs_mul = bs_plus * 0.5
        self.assertDictEqual(bs_mul.priorities, {'b': 1.0, 'd': 1.0, 'c': np.inf})

        bs3 = planning.BeliefState('c', {})
        bs_plus = bs1 + bs3
        self.assertDictEqual(bs_plus.priorities, bs1.priorities)
        
        bs_plus = bs3 + bs1
        self.assertDictEqual(bs_plus.priorities, bs1.priorities)
        
         

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
        cbs = planning.CBS(env)
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

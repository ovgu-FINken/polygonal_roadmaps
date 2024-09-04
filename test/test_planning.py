
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polygonal_roadmaps import cli
from polygonal_roadmaps.environment import GraphEnvironment, PlanningProblemParameters, gen_example_graph
import unittest
from hypothesis import given, strategies as st
import networkx as nx
from icecream import ic

from polygonal_roadmaps.planning import *

class TestPlans(unittest.TestCase):
    def test_plans(self):
        plans = Plans([[1,2,3], [1,2], [1]])
        self.assertListEqual(plans.plans, [[1,2,3], [1,2], [1]])
        self.assertListEqual(plans.as_state_list(), [(1,1,1), (2,2,None), (3,None,None)])
        plans2 = Plans.from_state_list([(1,1,1), (2,2,None), (3,None,None)])
        self.assertListEqual(plans.plans, plans2.plans)
    
    @given(x=st.lists(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=100)))
    def test_as_state_list(self, x):
        plans = Plans(x)
        state_list = plans.as_state_list()
        if len(x):
            self.assertListEqual([p[0] for p in plans], list(state_list[0]))
        plans2 = Plans.from_state_list(state_list)
        self.assertListEqual(plans.plans, plans2.plans)
        
    def test_validity_check(self):
        plans = Plans([[1,2,3], [1,2], [1]])
        self.assertFalse(plans.is_valid())
        
        self.assertTrue(Plans([[1,2,3], [4,5,1]]).is_valid())
        self.assertFalse(Plans([[1,2,3], [3,3,1]]).is_valid())
        
    def test_get_next_state(self):
        self.assertEqual(Plans([[1,2,3], [4,5,1]]).get_next_state(), (2,5))
        self.assertEqual(Plans([[1,2,3]]).get_next_state(), (2,))
        self.assertEqual(Plans([[1,2,3], [4], [5]]).get_next_state(), (2, None, None))
        

class TestPlanners(unittest.TestCase):
    def setUp(self):
        # simple graph, working
        self.env = GraphEnvironment(graph=gen_example_graph(5, 2), start=('b', 'g'), goal=('e', 'a'), planning_problem_parameters=PlanningProblemParameters(conflict_horizon=100))
        # more complex graph from swarmlab map
        self.env2 = cli.env_generator('DrivingSwarm;icra2021_map.yaml;icra2021.yml', n_agents=4)[0]
        self.env2.planning_problem_parameters = PlanningProblemParameters(conflict_horizon=100)
        # simple graph, not working (no path from start to goal for both agents without a conflict)
        self.env3 = GraphEnvironment(graph=gen_example_graph(5, 1), start=('a', 'e'), goal=('e', 'a'), planning_problem_parameters=PlanningProblemParameters(conflict_horizon=100))
        self.env4 = cli.env_generator('Graph;star.1', n_agents=2)[0]
  
    def check_env(self, Planner, env, **kwargs):
        planner = Planner(env, **kwargs)
        plan = planner.create_plan(env)
        self.assertTrue(plan.is_valid(env))

    def check_env_invalid(self, Planner, env, **kwargs):
        planner = Planner(env, **kwargs)
        with self.assertRaises(nx.NetworkXNoPath):
            plan = planner.create_plan(env)
            ic(Plans(plan).is_valid(env))

    def check_planner(self, Planner, **kwargs):
        self.check_env(Planner, self.env, **kwargs)
        self.check_env(Planner, self.env4, **kwargs)
        self.check_env(Planner, self.env2, **kwargs)
        #self.check_env_invalid(Planner, self.env3, **kwargs)
    
    def test_CCRv2(self):
        self.check_planner(CCRv2)
        
    def test_CCRv2_criticality(self):
        self.check_planner(CCRv2, quality_metric='criticality')

    def test_CCRv2_weighted_criticality(self):
        self.check_planner(CCRv2, quality_metric='weighted_criticality')

    def test_CBS(self):
        self.check_planner(CBSPlanner)

    def test_PrioSame(self):
        self.check_planner(PriorityAgentPlanner, priority_method='same')

    def test_PrioIndex(self):
        self.check_planner(PriorityAgentPlanner, priority_method='index')


    def test_StateValueAgent(self):
        self.check_planner(StateValueAgentPlanner)
        
    def test_ACO(self):
        self.check_planner(LearningAgentPlanner, method='aco')
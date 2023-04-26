import os
import sys
from pathlib import Path

from polygonal_roadmaps.cli import env_generator


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
from polygonal_roadmaps import planning, polygonal_roadmap, geometry
from polygonal_roadmaps.environment import MapfInfoEnvironment, RoadmapEnvironment, GraphEnvironment, Environment, PlanningProblemParameters
from polygonal_roadmaps.planning import CBSPlanner, FixedPlanner, PrioritizedPlanner, CCRPlanner


class TestPlanningExecution(unittest.TestCase):
    def setUp(self):
        self.envs = []
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        self.envs.append(MapfInfoEnvironment(scen_path, n_agents=4))
        self.envs.append(env_generator("DrivingSwarm;icra2021_map.yaml;icra2021.yml", n_agents=2)[0])

    def checkPlanner(self, planner_instance, *args, **kwargs):
        for env in self.envs:
            executor = polygonal_roadmap.Executor(env, planner_instance(env, *args, **kwargs))
            self.checkRun(executor)

    def checkRun(self, executor):
        executor.run()
        self.assertGreaterEqual(planning.compute_solution_robustness(executor.get_history_as_solution()),
                                1,
                                msg="Path should be k-robust with k>=1")
        self.assertEqual(len(executor.env.start), len(executor.history[0]))
        self.assertEqual(len(executor.env.start), len(executor.history[-1]))
        paths = executor.get_history_as_solution()
        self.assertTrue(planning.check_nodes_connected(executor.env.g, paths))

    def testGraphEnvironment(self):
        g = nx.from_edgelist([(1, 2), (2, 3), (1, 3), (1, 4)])
        env = GraphEnvironment(g, (1, 2), (2, 4))
        self.assertTrue(isinstance(env, Environment))

        planner = FixedPlanner(env, plan=[(1, 2), (2, 1), (2, 4)])

        executor = polygonal_roadmap.Executor(env, planner)
        self.assertListEqual(executor.history, [(1, 2)])
        executor.run()
        self.assertListEqual(executor.history, [(1, 2), (None, 1), (None, None)])
        executor.history = [(1, 2, 3), (1, 2, 3), (1, 2, 4), (1, 3, 4), (0, 3, 5)]
        partial_solution = executor.get_partial_solution()
        expected_solution = [[1, 0], [2], [3, 4, 5]]
        self.assertListEqual(partial_solution, expected_solution)

    def testRoadmapEnvironment(self):
        env = self.envs[1]
        prioritized_planner = PrioritizedPlanner(env)
        executor = polygonal_roadmap.Executor(env, prioritized_planner, time_frame=50)
        executor.run()
        # logging.warn(f'executer history: {executor.history}')
        self.assertGreaterEqual(planning.compute_solution_robustness(executor.get_history_as_solution()),
                                1,
                                msg="Path should be k-robust with k>=1")
        self.assertEqual(len(prioritized_planner.environment.start), len(executor.history[0]))
        self.assertEqual(len(prioritized_planner.environment.start), len(executor.history[-1]))
        
        paths = executor.get_history_as_solution()
        self.assertTrue(planning.check_nodes_connected(executor.env.g, paths))

    def testMapfInfoEnvironment(self):
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        env = MapfInfoEnvironment(scen_path, n_agents=2)
        self.assertIsInstance(env, Environment)

    def testPlanningWithCBS(self):
        self.checkPlanner(CBSPlanner)
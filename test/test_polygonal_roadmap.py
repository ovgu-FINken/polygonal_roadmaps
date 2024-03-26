import os
import sys
from pathlib import Path

from polygonal_roadmaps.cli import env_generator, save_run_data, load_run_data


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
import pandas as pd
import tempfile
from polygonal_roadmaps import executor, planning, geometry
from polygonal_roadmaps.environment import MapfInfoEnvironment, RoadmapEnvironment, GraphEnvironment, Environment, PlanningProblemParameters
from polygonal_roadmaps.planning import CBSPlanner, FixedPlanner, PrioritizedPlanner, CCRPlanner, PriorityAgentPlanner


class TestPlanningExecution(unittest.TestCase):
    def setUp(self):
        self.envs = []
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        self.envs.append(MapfInfoEnvironment(scen_path, n_agents=4))
        self.envs.append(env_generator("DrivingSwarm;icra2021_map.yaml;icra2021.yml", n_agents=2)[0])

    def checkPlanner(self, planner_instance, *args, **kwargs):
        for env in self.envs:
            exec = executor.Executor(env, planner_instance(env, *args, **kwargs))
            self.checkRun(exec)

    def checkRun(self, executor):
        executor.run()
        self.assertGreaterEqual(planning.compute_solution_robustness(executor.get_history_as_solution()),
                                1,
                                msg="Path should be k-robust with k>=1")
        self.assertEqual(len(executor.env.start), len(executor.history[0]))
        self.assertEqual(len(executor.env.start), len(executor.history[-1]))
        paths = executor.get_history_as_solution()
        self.assertEqual(executor.history[0], tuple(executor.env.start))
        self.assertTrue(planning.check_nodes_connected(executor.env.g, paths), f"nodes are not connected in plan: {paths}")

    def testGraphEnvironment(self):
        g = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 4), (4, 5)])
        env = GraphEnvironment(g, (1, 2), (2, 5))
        self.assertTrue(isinstance(env, Environment))

        planner = FixedPlanner(env, plan=[(1, 2), (1, 3), (2, 4), (None, 5)])

        exec = executor.Executor(env, planner)
        exec.replan = False
        # nothing happend yet, so history is empty
        # we want to save state and plans at that state
        self.assertListEqual(exec.history, [])
        exec.run()
        self.assertListEqual(exec.history, [(1, 2), (1, 3), (None, 4)])

    def testRoadmapEnvironment(self):
        env = self.envs[1]
        prioritized_planner = PriorityAgentPlanner(env, priority_method="index")
        exec = executor.Executor(env, prioritized_planner, time_frame=50)
        exec.run(replan=True)
        # logging.warn(f'executer history: {executor.history}')
        self.assertGreaterEqual(planning.compute_solution_robustness(exec.get_history_as_solution()),
                                1,
                                msg=f"Path should be k-robust with k>=1\nk={planning.compute_solution_robustness(exec.get_history_as_solution())}\n{exec.get_history_as_solution()}")
        self.assertEqual(len(prioritized_planner.environment.start), len(exec.history[0]))
        self.assertEqual(len(prioritized_planner.environment.start), len(exec.history[-1]))
        
        paths = exec.get_history_as_solution()
        self.assertTrue(planning.check_nodes_connected(exec.env.g, paths))

    def testMapfInfoEnvironment(self):
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        env = MapfInfoEnvironment(scen_path, n_agents=2)
        self.assertIsInstance(env, Environment)

    def testPlanningWithCBS(self):
        self.checkPlanner(CBSPlanner)
        
    def testLoadAndSave(self):
        test_data = {'a': 1, 'b': 2}
        test_df = pd.DataFrame([[1,2], [4,5], [7,9]], columns=['a', 'b'])
        with tempfile.TemporaryDirectory() as tempfile_path:
            print(tempfile_path)
            save_run_data(test_data, test_df, Path(tempfile_path))
            x, y = load_run_data(Path(tempfile_path))

        self.assertDictEqual(test_data, x)
        pd.testing.assert_series_equal(test_df['a'], y['a'], check_dtype=False)
        pd.testing.assert_series_equal(test_df['b'], y['b'], check_dtype=False)
        
import os
import sys
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
from polygonal_roadmaps import planner, polygonal_roadmap, geometry
from polygonal_roadmaps.environment import MapfInfoEnvironment, RoadmapEnvironment, GraphEnvironment, Environment
from polygonal_roadmaps.planner import Planner, CBSPlanner, FixedPlanner, PrioritizedPlanner, CCRPlanner


class TestPlanningExecution(unittest.TestCase):
    def setUp(self):
        self.envs = []
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        self.envs.append(MapfInfoEnvironment(scen_path, n_agents=4))
        map_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "icra2021_map.yaml"
        wx = (-1, 3.5)
        wy = (-3, 1.5)
        generators = geometry.square_tiling(0.5, working_area_x=wx, working_area_y=wy)
        self.envs.append(RoadmapEnvironment(map_path,
                                                              [26, 63],
                                                              [63, 26],
                                                              generator_points=generators,
                                                              wx=wx,
                                                              wy=wy,
                                                              offset=0.15))

    def checkPlanner(self, Planner, *args, **kwargs):
        for env in self.envs:
            executor = polygonal_roadmap.Executor(env, Planner(env, *args, **kwargs))
            self.checkRun(executor)

    def checkRun(self, executor):
        executor.run()
        self.assertGreaterEqual(planner.compute_solution_robustness(executor.get_history_as_solution()),
                                1,
                                msg="Path should be k-robust with k>=1")
        self.assertEqual(len(executor.env.start), len(executor.history[0]))
        self.assertEqual(len(executor.env.start), len(executor.history[-1]))
        paths = executor.get_history_as_solution()
        self.assertTrue(planner.check_nodes_connected(executor.env.g, paths))

    def testGraphEnvironment(self):
        g = nx.from_edgelist([(1, 2), (2, 3), (1, 3), (1, 4)])
        env = GraphEnvironment(g, (1, 2), (2, 4))
        self.assertTrue(isinstance(env, Environment))

        planner = FixedPlanner(env, [(1, 2), (2, 1), (2, 4)])
        self.assertIsInstance(planner, Planner)

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
        self.assertGreaterEqual(planner.compute_solution_robustness(executor.get_history_as_solution()),
                                1,
                                msg="Path should be k-robust with k>=1")
        self.assertEqual(len(prioritized_planner.env.start), len(executor.history[0]))
        self.assertEqual(len(prioritized_planner.env.start), len(executor.history[-1]))
        
        paths = executor.get_history_as_solution()
        self.assertTrue(planner.check_nodes_connected(executor.env.g, paths))

    def testMapfInfoEnvironment(self):
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        env = MapfInfoEnvironment(scen_path, n_agents=2)
        self.assertIsInstance(env, Environment)

    def testPlanningWithCBS(self):
        self.checkPlanner(CBSPlanner, limit=100)

    def testPlanningWithCBSHorizon(self):
        self.checkPlanner(CBSPlanner, horizon=3, discard_conflicts_beyond=3, limit=100)

    def testPlanningWithPrioritizedPlanner(self):
        self.checkPlanner(PrioritizedPlanner, limit=100)

    # def testPlanningWithPrioritizedPlannerHorizon(self):
    #     self.checkPlanner(polygonal_roadmap.PrioritizedPlanner, limit=100, discard_conflicts_beyond=30, horizon=30)

    def testPlanningCCRPlanner(self):
        self.checkPlanner(CCRPlanner, limit=100)

    def testPlanningCCRPlannerHSA(self):
        self.checkPlanner(CCRPlanner,
                          limit=100,
                          horizon=3,
                          discard_conflicts_beyond=3,
                          social_reward=0.1,
                          anti_social_punishment=0.1)
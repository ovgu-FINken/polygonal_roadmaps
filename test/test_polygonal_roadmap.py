import os
import sys
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import networkx as nx
from polygonal_roadmaps import polygonal_roadmap


class TestGraphCreation(unittest.TestCase):
    def testMinimalEnvironment(self):
        g = nx.from_edgelist([(1, 2), (2, 3), (1, 3), (1, 4)])
        env = polygonal_roadmap.GraphEnvironment(g, (1, 2), (2, 4))
        self.assertTrue(isinstance(env, polygonal_roadmap.Environment))

        planner = polygonal_roadmap.FixedPlanner(env, [(1, 2), (2, 1), (2, 4)])
        self.assertIsInstance(planner, polygonal_roadmap.Planner)

        executor = polygonal_roadmap.Executor(env, planner)
        self.assertListEqual(executor.history, [(1, 2)])
        executor.run(update=False)
        self.assertListEqual(executor.history, [(1, 2), (None, 1), (None, None)])

    def testMapfInfoEnvironment(self):
        map_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10.map"
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        env = polygonal_roadmap.MapfInfoEnvironment(map_path, scen_path, n_agents=2)
        self.assertIsInstance(env, polygonal_roadmap.Environment)

    def testPlanningWithCBS(self):
        map_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10.map"
        scen_path = Path(os.path.dirname(os.path.realpath(__file__))) / "resources" / "random-32-32-10-even-1.scen"
        env = polygonal_roadmap.MapfInfoEnvironment(map_path, scen_path, n_agents=2)
        planner = polygonal_roadmap.CBSPlanner(env)
        executor = polygonal_roadmap.Executor(env, planner)
        executor.run(update=False)
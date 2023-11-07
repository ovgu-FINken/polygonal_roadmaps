import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from polygonal_roadmaps.cli import run_scenario


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.scenarios = ["MAPF;room-32-32-4;even", "MAPF;empty-8-8;even", "DrivingSwarm;icra2021_map.yaml;icra2021.yml"]
        self.planners = ["CBS.yml", "PrioSame.yml", "PrioIndex.yml", "CCRv2.yml"]

    def integration_test(self, parameters=None):
        if parameters is None:
            parameters = "default.yml"
        for scenario in self.scenarios:
            for planner in self.planners:
                with self.subTest(msg=f"Test planning with {planner} in scenario {scenario} and parameters {parameters}."):
                    data = run_scenario(scenario,
                                  planner,
                                  n_agents=3,
                                  n_scenarios=1,
                                  problem_parameters="default.yml",
                                 )
                    assert data is not None
                    
    def test_default(self):
        self.integration_test()
    
    def test_horizon(self):
        self.integration_test(parameters="h5.yml")
    
    def test_random_progress(self):
        self.integration_test(parameters="s1.yml")

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from polygonal_roadmaps.cli import run_scenario


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.scenarios = ["MAPF;room-32-32-4;even", "MAPF;empty-8-8;even", "DrivingSwarm;icra2021_map.yaml;icra2021.yml"]
        self.planners = ["CBS.yml", "CCRv2.yml", "PrioSame.yml", "PrioIndex.yml"]
        self.problem_parameters = ["default.yml", "h5.yml", "s1.yml", "h5s1.yml"]

    def test_integration(self):
        for scenario in self.scenarios:
            for parameters in self.problem_parameters:
                for planner in self.planners:
                    with self.subTest(msg=f"Test planning with {planner} in scenario {scenario} and parameters {parameters}."):
                        data = run_scenario(scenario,
                                      planner,
                                      n_agents=3,
                                      n_scenarios=1,
                                      problem_parameters=parameters,
                                     )
                        assert data is not None

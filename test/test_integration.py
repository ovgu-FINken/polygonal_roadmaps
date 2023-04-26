import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from polygonal_roadmaps.cli import run_scenario


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.scenarios = ["MAPF;room-32-32-4;even", "MAPF;empty-8-8.map;even", "DrivingSwarm;icra2021_map.yaml;icra2021.yml"]
        self.planners = ["CCR.yml", "CBS.yml", "PRIO.yml"]
        self.problem_parameters = ["default.yml", "h3.yml"]
        #TODO: All the Planners


    def test_integration(self):
        for scenario in self.scenarios:
            for parameters in self.problem_parameters:
                for planner in self.planners:
                    with self.subTest(msg=f"Test planning with {planner} in scenario {scenario} and parameters {parameters}."):
                        data = run_scenario(scenario,
                                      planner,
                                      n_agents=3,
                                      n_scenarios=2,
                                      problem_parameters=parameters,
                                     )
                        assert data is not None

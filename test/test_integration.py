import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from polygonal_roadmaps.cli import run_scenarios


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.scenarios = ["room-32-32-4.map", "empty-8-8.map"]
        self.planners = ["CCR.yml", "CBS.yml"]


    def test_integration(self):
        for scenario in self.scenarios:
            for planner in self.planners:
                with self.subTest(msg=f"Test planning with {planner} in scenario {scenario}."):
                    data = run_scenarios(scenario,
                                  planner,
                                  n_agents=3,
                                  n_scenarios=2
                                 )
                    assert data is not None

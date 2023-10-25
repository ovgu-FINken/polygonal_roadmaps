
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from hypothesis import given, strategies as st

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
        
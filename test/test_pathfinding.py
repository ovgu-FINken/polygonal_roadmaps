import unittest


from polygonal_roadmaps import pathfinding

class TestCBS(unittest.TestCase):
    def testSumOfCost(self):
        result = pathfinding.sum_of_cost([[2], [2,3]])
        self.assertEqual(result, 3)

    def testPadPath(self):
        path = [1,2,3]
        path = pathfinding.pad_path(path, limit=100)
        self.assertEqual(len(path), 100)
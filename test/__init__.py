import unittest
import polygonal_roadmaps

def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(polygonal_roadmaps)
    return suite

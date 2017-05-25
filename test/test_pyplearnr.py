import sys

sys.path.append("../code")

import pyplearnr as ppl

import unittest

class AugmentedTestCase(unittest.TestCase):
    """
    unittest.TestCase class with an extra helper method for comparing expected
    and actual errors
    """
    def assert_with_messsage(self, msg, func, args, kwargs):
        try:
            func(*args, **kwargs)
            # self.assertFail()
        except Exception as inst:
            self.assertEqual(inst.message, msg)

class NestedKFoldCrossValidationTestCase(AugmentedTestCase):
    """
    Tests NestedKFoldCrossValidation class
    """
    def test_init_outer_loop_fold_count_zero(self):
        ############### Test initialization inputs ###############
        msg = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer"

        kwargs = {
            'outer_loop_fold_count': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [],kwargs)

    def test_init_outer_loop_fold_count_negative(self):
        ############### Test initialization inputs ###############
        msg = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer"

        kwargs = {
            'outer_loop_fold_count': -5
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [],kwargs)

    def test_init_inner_loop_fold_count_zero(self):
        msg = "The inner_loop_fold_count" \
            " keyword argument, dictating the number of folds in the inner" \
            " loop, must be a positive integer"

        kwargs = {
            'inner_loop_fold_count': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)

    def test_init_inner_loop_fold_count_negative(self):
        msg = "The inner_loop_fold_count" \
            " keyword argument, dictating the number of folds in the inner" \
            " loop, must be a positive integer"

        kwargs = {
            'inner_loop_fold_count': -5
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)


    def test_init_outer_loop_split_seed_zero(self):
        msg = "The " \
            "outer_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the outer loop, must be an integer."

        kwargs = {
            'outer_loop_split_seed': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)

    def test_init_outer_loop_split_seed_negative(self):
        msg = "The " \
            "outer_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the outer loop, must be an integer."

        kwargs = {
            'outer_loop_split_seed': -5
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)

    def test_init_inner_loop_split_seed_zero(self):
        msg = "The " \
            "inner_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the inner loop, must be an integer."

        kwargs = {
            'inner_loop_split_seed': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)

    def test_init_inner_loop_split_seed_negative(self):
        msg = "The " \
            "inner_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the inner loop, must be an integer."

        kwargs = {
            'inner_loop_split_seed': -5
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [], kwargs)

if __name__ == '__main__':
    unittest.main()

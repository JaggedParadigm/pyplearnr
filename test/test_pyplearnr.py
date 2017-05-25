import sys

sys.path.append("../code")

import pyplearnr as ppl

import unittest

class NestedKFoldCrossValidationTestCase(unittest.TestCase):
    """
    Tests NestedKFoldCrossValidation class
    """
    def test_init_outer_loop_fold_count(self):
        ############### Test initialization inputs ###############
        msg = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer"

        kwargs = {
            'outer_loop_fold_count': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [],kwargs)

    def test_init_inner_loop_fold_count(self):
        print 'ayy'
        msg = "The inner_loop_fold_count" \
            " keyword argument, dictating the number of folds in the inner" \
            " loop, must be a positive integer"

        kwargs = {
            'inner_loop_fold_count': 0
        }

        self.assert_with_messsage(msg, ppl.NestedKFoldCrossValidation,
                                  [],kwargs)


    def assert_with_messsage(self, msg, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            self.assertFail()
        except Exception as inst:
            self.assertEqual(inst.message, msg)

if __name__ == '__main__':
    unittest.main()

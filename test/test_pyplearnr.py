import sys

sys.path.append("../code")

import pyplearnr as ppl

import pandas as pd

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

    def test_get_outer_split_indices(self):
        # Read data into Pandas dataframe
        df = pd.read_pickle('../trimmed_titanic_data.pkl')

        simulation_df = df.copy()

        # Set categorial features as such
        categorical_features = ['Survived','Pclass','Sex','Embarked','Title']

        for feature in categorical_features:
            simulation_df[feature] = simulation_df[feature].astype('category')

        # One-hot encode categorical features
        simulation_df = pd.get_dummies(simulation_df,drop_first=True)

        output_feature = 'Survived_1'

        column_names = list(simulation_df.columns)

        input_features = [x for x in column_names if x != output_feature]

        # Split into features and targets
        X = simulation_df[input_features].copy().values
        y = simulation_df[output_feature].copy().values

        # Obtain test/train split indices for outer and inner folds
        kfcv = ppl.NestedKFoldCrossValidation()

        kfcv.get_outer_split_indices(X, y=y, stratified=False)

        # Test that the resulting indices combine to form the total set of
        # indices
        outer_test_inds_target = set(range(X.shape[0]))

        all_outer_test_inds = set()

        for outer_fold_ind, outer_fold in kfcv.outer_folds.iteritems():
            current_outer_test_fold_inds = outer_fold.test_fold_inds
            current_outer_train_fold_inds = outer_fold.train_fold_inds

            all_outer_test_inds |= set(current_outer_test_fold_inds)

            inner_test_inds_target = set(range(X[current_outer_train_fold_inds].shape[0]))

            all_inner_test_inds = set()
            for inner_fold_ind, inner_fold in outer_fold.inner_folds.iteritems():
                all_inner_test_inds |= set(inner_fold.test_fold_inds)

            self.assertTrue(not all_inner_test_inds-inner_test_inds_target)

        self.assertTrue(not all_outer_test_inds-outer_test_inds_target)





if __name__ == '__main__':
    unittest.main()

# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

import sys

sys.path.append("../code")

import pyplearnr as ppl

import pandas as pd

import itertools

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

    def get_cleaned_titanic_data(self):
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

        return X, y

class PipelineBundleTestCase(AugmentedTestCase):
    """
    Tests PipelineBundle methods
    """
    def test_build_bundle(self):
        # Set test pipeline bundle schematic
        pipeline_bundle_schematic = [
            {'scaler': {
                'standard': {},
                'normal': {}
            }},
            {'estimator': {
                'knn': {
                    'n_neighbors': range(1,11),
                    'weights': ['uniform', 'distance']
                },
                'svm': {
                    'C': range(1,12)
                }
            }}
        ]

        pipelines = ppl.PipelineBundle().build_pipeline_bundle(pipeline_bundle_schematic)



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
        # Get data fit for testing
        X, y = self.get_cleaned_titanic_data()

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

    def test_fit(self):
        # Get data fit for testing
        X, y = self.get_cleaned_titanic_data()

        # Obtain test/train split indices for outer and inner folds
        kfcv = ppl.NestedKFoldCrossValidation()

        estimators = ['logistic_regression','svm']

        # feature_interaction_options = [True,False]
        feature_selection_options = [None,'select_k_best']
        scaling_options = [None,'standard','normal','min_max','binary']
        transformations = [None,'pca']

        pipeline_steps = [feature_selection_options,scaling_options,
                          transformations,estimators]

        pipeline_options = list(itertools.product(*pipeline_steps))






        kfcv.fit(X, y, pipelines=[], stratified=True)

        """
        best_pipeline = {
            "trained_all_pipeline": None,
            "mean_validation_score": None,
            "validation_score_std": None
        }

        |   *    |           |           |
            best_outer_fold_1_pipeline = {
                "outer_fold_id": None
                "best_pipeline_ind": None,
                "trained_all_best_pipeline": None,
                "validation_score": None,
                "scoring_type": None
            }
            pipeline_1_outer_fold_1 = {
                "id": None,
                "mean_test_score": None,
                "test_score_std": None,
                "mean_train_score": None,
                "train_score_std": None,
                "scoring_type": None
            }
            pipeline_2_outer_fold_1
            ....
            pipeline_d_outer_fold_1

                |   *  |      |         |
                    pipeline_1_outer_fold_1_inner_fold_1 = {
                        'id': None,
                        'outer_fold_id': None,
                        'inner_fold_id': None,
                        'pipeline': None,
                        'test_score': None,
                        'train_score': None,
                        'scoring_type': None,
                    }
                    pipeline_2_outer_fold_1_inner_fold_1
                    ....
                    pipeline_d_outer_fold_1_inner_fold_1
                |      |   *  |         |
                    pipeline_1_outer_fold_1_inner_fold_2
                    pipeline_2_outer_fold_1_inner_fold_2
                    ....
                    pipeline_d_outer_fold_1_inner_fold_2
                |      |      |    *    |
                    pipeline_1_outer_fold_1_inner_fold_3
                    pipeline_2_outer_fold_1_inner_fold_3
                    ....
                    pipeline_d_outer_fold_1_inner_fold_3
        |       |      *     |            |
        ............
        |       |            |     *      |
        ............
        """

        """
        Alternate setup:

        'scoring_metric': None,
        best_pipeline = {
            "trained_all_pipeline": None,
            "mean_validation_score": None,
            "validation_score_std": None
        },
        'outer_folds' = {
            'id': None,
            'test_inds': None,
            'train_inds': None,

            'best_pipeline': {
                'best_pipeline_validation_score': None,

            },





            'pipelines': {
                'id': {
                    'id': None
                    'mean_test_score': None,
                    'test_score_std': None,
                    'pipeline': None
                }
            }

            'inner_folds': {
                'id': None,
                'test_fold_inds': None,
                'train_fold_inds': None,
                'pipelines': {
                    'id': {
                        'id': outer_inner_pipeline
                        'test_score': None,
                        'train_score': None,
                        'pipeline': None
                    }
                }


            },
            ...
            {},
        }

        """







if __name__ == '__main__':
    unittest.main()

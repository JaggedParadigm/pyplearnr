# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

import numpy as np

from sklearn.base import clone

from .pipeline_evaluator import PipelineEvaluator

class Fold(object):
    """
    Class containing test/train split indices for inner folds of nest k-fold
    cross-validation run
    """
    def __init__(self, fold_id=None, test_fold_inds=None,
                 train_fold_inds=None):
        ############### Initialize fields ###############
        self.fold_id = fold_id

        self.test_fold_inds = test_fold_inds

        self.train_fold_inds = train_fold_inds

        self.X_test = None
        self.y_test = None

        self.X_train = None
        self.y_train = None

        self.pipelines = {}

    def fit(self, outer_loop_X_train, outer_loop_y_train, pipelines,
            scoring_metric='auc'):
        """
        Fits the pipeslines to the current inner fold training data
        """
        self.X_test = outer_loop_X_train[self.test_fold_inds]
        self.y_test = outer_loop_y_train[self.test_fold_inds]

        self.X_train = outer_loop_X_train[self.train_fold_inds]
        self.y_train = outer_loop_y_train[self.train_fold_inds]

        for pipeline_id, pipeline in pipelines.iteritems():
            # Initialize this combination
            self.pipelines[pipeline_id] = {
                'id': pipeline_id,

                'pipeline': clone(pipeline, safe=True),

                'y_test_pred': None,
                'y_train_pred': None,

                'test_score': None,
                'train_score': None,

                'classification_report': None
            }

            # Fit pipeline to training set of fold
            self.pipelines[pipeline_id]['pipeline'].fit(self.X_train,
                                                        self.y_train
                                                        )

            # Calculate predicted targets from input test data
            self.pipelines[pipeline_id]['y_test_pred'] = \
                self.pipelines[pipeline_id]['pipeline'].predict(self.X_test)

            # Calculate the training prediction for this model
            self.pipelines[pipeline_id]['y_train_pred'] = \
                self.pipelines[pipeline_id]['pipeline'].predict(self.X_train)

            # Calculate train score
            self.pipelines[pipeline_id]['train_score'] = \
                PipelineEvaluator().get_score(
                    self.y_train,
                    self.pipelines[pipeline_id]['y_train_pred'],
                    scoring_metric
                    )

            # Calculate test score
            self.pipelines[pipeline_id]['test_score'] = \
                PipelineEvaluator().get_score(
                    self.y_test,
                    self.pipelines[pipeline_id]['y_test_pred'],
                    scoring_metric
                    )
                    
class OuterFold(Fold):
    """
    Class containing test/train split indices for data
    """
    def __init__(self, fold_id=None, test_fold_inds=None,
                 train_fold_inds=None):
        ############### Initialize fields ###############
        # Tell class it is itself for weird Jupyter notebook %autoload
        # incompatibility
        self.__class__ = OuterFold

        super(OuterFold, self).__init__(fold_id=fold_id,
                                        test_fold_inds=test_fold_inds,
                                        train_fold_inds=train_fold_inds)

        # Folds for inner k-fold cross-validation
        self.inner_folds = {}

        self.best_pipeline_ind = None

    def fit(self, shuffled_X, shuffled_y, pipelines, scoring_metric='auc'):
        """
        Performs inner loop of nested k-fold cross-validation for current outer
        fold and returns the winner's index
        """
        # Save data
        self.X_test = shuffled_X[self.test_fold_inds]
        self.y_test = shuffled_y[self.test_fold_inds]

        self.X_train = shuffled_X[self.train_fold_inds]
        self.y_train = shuffled_y[self.train_fold_inds]

        # Save pipelines

        # Fit all pipelines to the training set of each inner fold and Calculate
        # inner training and test scores
        for inner_fold_ind, inner_fold in self.inner_folds.iteritems():
            inner_fold.fit(self.X_train, self.y_train, pipelines,
                           scoring_metric=scoring_metric)

        # Calculate and save statistics for train/test fold scores for each
        # pipeline and find the that withthe maximum median
        max_score = -1e14
        max_ind = -1
        for pipeline_id, pipeline in pipelines.iteritems():
            # Initialize pipeline
            self.pipelines[pipeline_id] = {
                'id': pipeline_id,

                'pipeline': clone(pipeline, safe=True),
                'all_trained_pipeline': None,

                'y_validation_pred': None,
                'outer_y_train_pred': None,

                'outer_train_score': None,
                'validation_score': None,

                'mean_inner_test_score': None,
                'median_inner_test_score': None,
                'inner_test_score_std': None,

                'mean_inner_train_score': None,
                'median_inner_train_score': None,
                'inner_train_score_std': None,





                'test_classification_report': None,

                'scoring_metric': None
            }

            # Collect test and train scores
            test_scores = []
            train_scores = []
            for inner_fold_ind, inner_fold in self.inner_folds.iteritems():
                test_scores.append(
                    inner_fold.pipelines[pipeline_id]['test_score'])
                train_scores.append(
                    inner_fold.pipelines[pipeline_id]['train_score']
                    )

            # Calculate and save statistics on test and train scores
            self.pipelines[pipeline_id]['mean_test_score'] = \
                np.mean(test_scores)
            self.pipelines[pipeline_id]['median_test_score'] = \
                np.median(test_scores)
            self.pipelines[pipeline_id]['test_score_std'] = np.std(test_scores,
                                                                   ddof=1)

            self.pipelines[pipeline_id]['mean_train_score'] = \
                np.mean(train_scores)
            self.pipelines[pipeline_id]['median_train_score'] = \
                np.median(train_scores)
            self.pipelines[pipeline_id]['train_score_std'] = \
                np.std(train_scores,ddof=1)

            # Find highest score and corresponding pipeline index
            if max_score < self.pipelines[pipeline_id]['median_test_score']:
                max_score = self.pipelines[pipeline_id]['median_test_score']
                max_ind = pipeline_id

        self.best_pipeline_ind = max_ind

        best_pipeline = self.pipelines[self.best_pipeline_ind]

        # Initialize pipeline that will be trained on all outer-fold training
        # data
        best_pipeline['all_trained_pipeline'] = clone(pipelines[max_ind],
                                                       safe=True)

        best_pipeline['scoring_metric'] = scoring_metric

        # Train on all inner loop training data
        best_pipeline['all_trained_pipeline'].fit(self.X_train, self.y_train)

        # Form predictions for validation and training targets
        best_pipeline['y_validation_pred'] = \
            best_pipeline['all_trained_pipeline'].predict(self.X_test)
        best_pipeline['outer_y_train_pred'] = \
            best_pipeline['all_trained_pipeline'].predict(self.X_train)

        # Form actually and predicted target comparison pairs
        train_comparison_pair = [self.y_train,
                                 best_pipeline['outer_y_train_pred']]
        test_comparison_pair = [self.y_test,
                                best_pipeline['y_validation_pred']]

        # Calculate outer loop training score
        best_pipeline['outer_train_score'] = \
            PipelineEvaluator().get_score(*train_comparison_pair + [scoring_metric])

        # Calculate validation score

        best_pipeline['validation_score'] = \
            PipelineEvaluator().get_score(*test_comparison_pair + [scoring_metric])

        # Calculate classification report
        best_pipeline['test_classification_report'] = \
            PipelineEvaluator().get_classification_report(
                *test_comparison_pair
                )

    def train_winning_pipeline(self, winning_pipeline_ind, scoring_metric):
        """
        Trains pipeline, corresponding to a user-provided key, on training
        data, and scores it on the testing data if a test score isn't found.
        """
        winning_pipeline = self.pipelines[winning_pipeline_ind]

        validation_score = winning_pipeline['validation_score']

        if not validation_score:
            winning_pipeline['all_trained_pipeline'] = clone(
                winning_pipeline['pipeline']
                )

            # Train on all inner loop training data
            winning_pipeline['all_trained_pipeline'].fit(self.X_train,
                                                         self.y_train)

            # Form predictions for validation and training targets
            winning_pipeline['y_validation_pred'] = \
                winning_pipeline['all_trained_pipeline'].predict(self.X_test)
            winning_pipeline['outer_y_train_pred'] = \
                winning_pipeline['all_trained_pipeline'].predict(self.X_train)

            # Form actually and predicted target comparison pairs
            train_comparison_pair = \
                [self.y_train, winning_pipeline['outer_y_train_pred']]
            test_comparison_pair = \
                [self.y_test, winning_pipeline['y_validation_pred']]

            # Calculate outer loop training score
            winning_pipeline['outer_train_score'] = \
                PipelineEvaluator().get_score(
                    *train_comparison_pair + [scoring_metric]
                    )

            # Calculate validation score
            winning_pipeline['validation_score'] = \
                PipelineEvaluator().get_score(
                    *test_comparison_pair + [scoring_metric]
                    )

            winning_pipeline['test_classification_report'] = \
                PipelineEvaluator().get_classification_report(
                    *test_comparison_pair
                    )

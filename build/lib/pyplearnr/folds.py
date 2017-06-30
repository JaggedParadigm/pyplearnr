# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np

from sklearn.base import clone

from .trained_pipeline import TrainedPipeline, OuterFoldTrainedPipeline

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
            scoring_metric='auc', score_type='median'):
        """
        Fits the pipeslines to the current inner fold training data
        """
        self.X_test = outer_loop_X_train[self.test_fold_inds]
        self.y_test = outer_loop_y_train[self.test_fold_inds]

        self.X_train = outer_loop_X_train[self.train_fold_inds]
        self.y_train = outer_loop_y_train[self.train_fold_inds]

        for pipeline_id, pipeline in pipelines.items():
            pipeline_kwargs = {
            'pipeline_id': pipeline_id,
            'pipeline': clone(pipeline, safe=True),
            'scoring_metric': scoring_metric,
            'score_type': score_type
            }

            # Initialize this combination
            self.pipelines[pipeline_id] = TrainedPipeline(**pipeline_kwargs)

            # Fit pipeline to training set of fold
            self.pipelines[pipeline_id].fit(self.X_train, self.y_train,
                                            self.X_test, self.y_test)

    def get_pipeline_scores(self, pipeline_id=None, fold_type=None):
        """
        Returns either test or train score of desired pipeline

        Parameters
        ----------
        pipeline_id :   int
            External id of pipeline

        fold_type : str, {'test', 'train'}
            Type of score to obtain for designated pipeline
            test: Score(s) for test fold
            train: Score(s) for train fold

        Returns
        -------
        scores : list of floats
            Scores corresponding to the test or train fold of the desired
            pipeline
        """
        ############### Check inputs ###############
        if type(pipeline_id) is not int:
            raise Exception("The desired pipeline index, pipline_ind, must " \
                            "be provided and must be of type int")

        if not fold_type or fold_type not in ['test', 'train']:
            raise Exception("The fold type, fold_type, keyword argument " \
                            "must be provide and either 'test' or 'train'")

        ############### Obtain inner fold scores ###############
        if fold_type == 'test':
            scores = self.pipelines[pipeline_id].test_scores
        elif fold_type == 'train':
            scores = self.pipelines[pipeline_id].train_scores

        return scores

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

    def fit(self, shuffled_X, shuffled_y, pipelines, scoring_metric='auc',
            score_type='median', tie_breaker='choice',
            best_inner_fold_pipeline_ind=None):
        """
        Performs inner loop of nested k-fold cross-validation for current outer
        fold

        Parameters
        ----------
        shuffled_X :    numpy.ndarray, shape (m, n)
            Feature value matrix with each row corresponding to a
            one set of feature values and each column corresponding
            to a particular feature

        shuffled_y :    numpy.ndarray, shape (m, )
            Target value vector

        pipelines :     dict of sklearn.pipeline.Pipeline objects with integer
                        keys
            Labeled scikit-learn pipelines

        scoring_metric :    str, {'auc'}
            Metric used to score estimator

        """
        # Form keyword arguments needed to choose and train the best
        # pipeline
        choose_best_inner_fold_pipeline_kwargs = {
            'score_type': score_type,
            'tie_breaker': tie_breaker
        }

        ############### Save data and train inner pipelines ###############
        if not best_inner_fold_pipeline_ind:
            ############### Form and save test/train fold data ###############
            # Save data
            self.X_test = shuffled_X[self.test_fold_inds]
            self.y_test = shuffled_y[self.test_fold_inds]

            self.X_train = shuffled_X[self.train_fold_inds]
            self.y_train = shuffled_y[self.train_fold_inds]

            ############### Fit inner fold pipelines ###############
            # Fit all pipelines to the training set of each inner fold and
            # calculate inner training and test scores
            self.fit_inner_fold_pipelines(pipelines,
                                          scoring_metric=scoring_metric,
                                          score_type=score_type)

            # Calculate and save statistics for train/test fold scores for each
            # pipeline and find the that with the maximum median
            self.collect_inner_loop_scores(pipelines,
                                           scoring_metric=scoring_metric,
                                           score_type=score_type)

        ############### Choose best inner fold pipeline ###############
        self.choose_best_inner_fold_pipeline(
            score_type=score_type,
            tie_breaker=tie_breaker,
            best_inner_fold_pipeline_ind=best_inner_fold_pipeline_ind)

    def train_winning_pipeline(self, best_pipeline_ind):
        """
        Trains ultimate winner of the inner-loop of the nested k-fold cross-
        validation contest on the training set of this outer loop and scores it
        based on the previously saved scoring metric.
        """
        self.pipelines[best_pipeline_ind].fit(self.X_train, self.y_train,
                                              self.X_test, self.y_test)

    def choose_best_inner_fold_pipeline(self, score_type='median',
                                        tie_breaker='choice',
                                        best_inner_fold_pipeline_ind=None):
        """
        Chooses the winner of the inner-loop folds based on highest score. If
        there are multiple winners, the user can specify a winner
        (preferably the simplest pipeline/model) by using the
        train_best_inner_fold_pipeline method.

        Parameters
        ----------
        score_type :    str, {'mean', 'median'}, optional
            Statistical measure used to pick the winner(s) of the inner-loop
            contest
            mean :  Use the highest mean of the inner-fold test scores to pick
                    the best pipeline/model
            median :    Use the highest median of the inner-fold test scores to
                        pick the best pipeline/model

        tie_breaker :   str, {'choice', 'first'}
            Decision rule to use to decide the winner in the event of a tie
            choice :    Inform the user that they need to use the
                        choose_best_pipelines method to pick the winner of the
                        inner loop contest
            first :     Simply use the first model with the same score

        """
        best_pipeline_ind = None

        if best_inner_fold_pipeline_ind is None:
            ############### Check inputs ###############
            if score_type not in ['mean', 'median']:
                raise Exception("The keyword argument used to designate the " \
                                "metric used to judge models, 'score_type', " \
                                "must be either 'mean' or 'median'")

            if tie_breaker not in ['choice', 'first']:
                raise Exception("The keyword argument dictating the decision " \
                                "rule to be used in case of a tie between " \
                                "pipelines in the inner loop, 'tie_breaker', " \
                                "must be either 'choice' or 'first'")

            ############### Find prospective winners ###############
            best_pipeline_inds = \
                self.choose_best_pipelines(score_type=score_type)

            ############### Train best pipeline or resolve tie ###############
            # return best_pipeline_inds
            if len(best_pipeline_inds) == 1:
                # Save winner if only one best_pipeline_ind, outer_fold_ind
                best_pipeline_ind = best_pipeline_inds[0]
            else:
                if tie_breaker=='choice':
                    # Encourage user to choose simplest model if there is no clear
                    # winner
                    print("Outer Fold: %d"%(self.fold_id), '\n')
                    for pipeline_ind in best_pipeline_inds:
                        print(pipeline_ind, self.pipelines[pipeline_ind].pipeline)
                    print("\n\nNo model was chosen because there is no clear " \
                          "winner. Please use the same fit method with " \
                          "best_inner_fold_pipeline_inds keyword argument." \
                          "\n\nExample:\tkfcv.fit(X.values, y.values, " \
                          "pipelines)\n\t\tkfcv.fit(X.values, y.values, " \
                          "pipelines, \n\t\t\t best_inner_fold_pipeline_inds = "\
                          "{0:9, 2:3})\n")
                elif tie_breaker == 'first':
                    best_pipeline_ind = best_pipeline_inds[0]
        else:
            best_pipeline_ind = best_inner_fold_pipeline_ind

        if best_pipeline_ind is not None:
            self.best_pipeline_ind = best_pipeline_ind

    def choose_best_pipelines(self, score_type=None):
        """
        Chooses pipeline with the highest test score

        Parameters
        ----------
        score_type :    str, {'mean', 'median'}, optional
            Statistical measure to return
            mean :  Use the highest mean of the inner-fold test scores to pick
                    the best pipeline/model
            median :    Use the highest median of the inner-fold test scores to
                        pick the best pipeline/model

        Returns
        -------
        winning_pipeline_ind :  int
            Index of pipeline with highest score

        """
        # Collect scores for each pipeline
        centrality_measures = {}
        for pipeline_ind, pipeline in self.pipelines.items():
             centrality_measures[pipeline_ind] = \
                pipeline.get_inner_loop_score_center(score_type=score_type,
                                                     fold_type='test')

        # Find maximum score and corresponding index
        max_score = max([score for x, score in centrality_measures.items()])


        winner_inds = [pipeline_ind for pipeline_ind, score \
                     in centrality_measures.items() if score==max_score]

        return winner_inds

    def collect_inner_loop_scores(self, pipelines, scoring_metric=None,
                                  score_type=None):
        """
        Initializes outer loop fold-associated pipelines and collects inner
        fold scores for the same pipeline

        Parameters
        ----------
        pipelines : dict of pyplearnr.OuterFoldTrainedPipelines
            Labeled pipeslines to obtain all inner-fold scores from

        """
        for pipeline_id, pipeline in pipelines.items():
            pipeline_kwargs = {
            'pipeline_id': pipeline_id,
            'pipeline': clone(pipeline, safe=True),
            'scoring_metric': scoring_metric,
            'score_type': score_type
            }

            # Initialize pipeline
            self.pipelines[pipeline_id] = OuterFoldTrainedPipeline(
                                            **pipeline_kwargs
                                            )

            # Collect test and train scores for current pipeline in inner folds
            # and save
            test_scores = []
            train_scores = []
            for inner_fold in self.inner_folds.values():
                test_scores.append(
                    inner_fold.get_pipeline_scores(pipeline_id=pipeline_id,
                                                   fold_type='test')[0])
                train_scores.append(
                    inner_fold.get_pipeline_scores(pipeline_id=pipeline_id,
                                                   fold_type='train')[0])

            self.pipelines[pipeline_id].set_inner_loop_scores(train_scores,
                                                              test_scores)

    def fit_inner_fold_pipelines(self, pipelines, scoring_metric=None,
                                 score_type=None):
        """
        Fit inner folds of current outer fold training set to all provided
        pipelines.

        Parameter
        ---------
        pipelines : dict of sklearn.pipeline.Pipeline objects with integer keys
            Labeled scikit-learn pipelines

        scoring_metric :    str, {'auc', 'rmse'}, optional
            Scoring metric used to score pipelines/models. Used to figure out
            if a classifier or regressor is at the end of the pipeline
            auc :   Area under the ROC curve
            rmse :  Root mean-squared error

        score_type :    str, {'mean', 'median'}, optional
            Statistical category to use to compare models and choose the best
            mean :  Use the highest mean of the inner-fold test scores to pick
                    the best pipeline/model
            median :    Use the highest median of the inner-fold test scores to
                        pick the best pipeline/model

        """
        for inner_fold in self.inner_folds.values():
            inner_fold.fit(self.X_train, self.y_train, pipelines,
                           scoring_metric=scoring_metric, score_type=score_type)

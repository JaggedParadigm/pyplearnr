# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Python 2/3 compatibility
from __future__ import print_function

# Basic tools
import numpy as np
import pandas as pd
import random
import re

# For scikit-learn pipeline cloning
from sklearn.base import clone

# Graphing
import pylab as plt

import matplotlib
import matplotlib.pyplot as mpl_plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as cmx

# Cross validation tools
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Other pyplearnr classes
from .folds import Fold, OuterFold

from .trained_pipeline import OuterFoldTrainedPipeline

from .pipeline_builder import PipelineBuilder

class NestedKFoldCrossValidation(object):
    """
    Class that handles nested k-fold cross validation, whereby the inner loop
    handles the model selection and the outer loop is used to provide an
    estimate of the chosen model/pipeline's out of sample score.
    """
    def __init__(self, outer_loop_fold_count=3, inner_loop_fold_count=3,
                 outer_loop_split_seed=None, inner_loop_split_seeds=None,
                 shuffle_seed=None, shuffle_flag=True,
                 random_combinations=None, random_combination_seed=None,
                 schematic=None):
        """
        Parameters
        ----------
        outer_loop_fold_count : int, optional
            Number of folds in the outer-loop of the nested k-fold
            cross-validation.

        inner_loop_fold_count : int, optional
            Number of folds in the inner loops of the nested k-fold
            cross-validation for each outer-fold.

        outer_loop_split_seed : int, optional
            Seed determining how the data will be split into outer folds. This,
            along with the other two seeds should be sufficient to reproduce
            results.

        inner_loop_split_seeds :    list of int, optional
            Seeds determining how the training data of the outer folds will be
            split. These, along with the other two seeds should be sufficient
            to reproduce results.

        shuffle_seed :  int, optional
            Seed determining how the data will be shuffled if the shuffle_flag
            is set to True. This, along with the other two seeds should be
            sufficient to reproduce results.

        shuffle_flag :  boolean, optional
            Determines whether the data will be shuffled.
            True :  Shuffle data and store shuffled data and the indices used
                    to generate it using the shuffle_seed. Seed is randomly
                    assigned if not provided.
            False : Data is not shuffled and indices that, if used as indices
                    for the data will result in the same data, are saved.

        """
        ############### Save initial inputs ###############
        self.shuffle_flag = shuffle_flag

        self.shuffle_seed = shuffle_seed

        self.outer_loop_fold_count = outer_loop_fold_count
        self.inner_loop_fold_count = inner_loop_fold_count

        self.outer_loop_split_seed = outer_loop_split_seed

        self.inner_loop_split_seeds = inner_loop_split_seeds

        self.random_combinations = random_combinations
        self.random_combination_seed = random_combination_seed

        ############### Initialize other fields ###############
        # Shuffled data indices
        self.shuffled_data_inds = None

        # OuterFold objects fold indices
        self.outer_folds = {}

        # Input data and targets
        self.X = None
        self.y = None

        self.shuffled_X = None
        self.shuffled_y = None

        self.pipelines = None

        # Best pipeline trained on all data
        self.pipeline = None

        # Essentially, uses this measure of centrality (mean or median) of the
        # inner-fold scores to decide winning for that outer-fold
        self.score_type = None

        # Metric to calculate as the pipeline scores (Ex: 'auc', 'rmse')
        self.scoring_metric = None

        self.best_pipeline_ind = None

        ############### Populate fields with defaults ###############
        if self.shuffle_seed is None:
            self.shuffle_seed = random.randint(1,5000)

        # Generate seeds if not given (*200 since 5-fold CV results in range
        # of 1 to 1000)
        if self.outer_loop_split_seed is None:
            self.outer_loop_split_seed = random.randint(
                                            1,
                                            self.outer_loop_fold_count*200)

        if self.inner_loop_split_seeds is None:
            self.inner_loop_split_seeds = np.random.randint(
                                            1,
                                            high=self.inner_loop_fold_count*200,
                                            size=self.outer_loop_fold_count)

        if self.random_combinations is not None:
            if self.random_combination_seed is None:
                self.random_combination_seed = random.randint(1,5000)

        ############### Check fields so far ###############
        outer_loop_fold_count_error = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer."

        assert type(self.outer_loop_fold_count) is int, \
            outer_loop_fold_count_error

        assert self.outer_loop_fold_count > 0, outer_loop_fold_count_error

        inner_loop_fold_count_error = "The inner_loop_fold_count" \
            " keyword argument, dictating the number of folds in the inner" \
            " loop, must be a positive integer"

        assert type(self.inner_loop_fold_count) is int, inner_loop_fold_count_error

        assert self.inner_loop_fold_count > 0, inner_loop_fold_count_error

        assert type(self.outer_loop_split_seed) is int, "The " \
            "outer_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the outer loop, must be an integer."

        if type(self.inner_loop_split_seeds) is not np.ndarray \
            and type(self.inner_loop_split_seeds) is not list:
            raise Exception("The inner_loop_split_seed keyword argument," \
            " dictating how the data is split into folds for the inner"\
            " loop, must be of type np.ndarray or list" )

        assert len(self.inner_loop_split_seeds) == self.outer_loop_fold_count, \
            "The number of inner-loop contest seeds must be equal to the " \
            "number of outer-folds"

        if self.random_combinations is not None:
            assert type(self.random_combinations) is int, "The number of " \
                "pipeline step/parameter combinations, random_combinations," \
                " must be of type int."

        if self.random_combination_seed is not None:
            assert type(self.random_combination_seed) is int, "The seed " \
                "determining how the exact pipeline step/parameter " \
                "combinations is chosen, random_combination_seed," \
                " must be of type int."

    def fit(self, X, y, pipelines=None, stratified=True, scoring_metric='auc',
            tie_breaker='choice', best_inner_fold_pipeline_inds=None,
            best_outer_fold_pipeline=None, score_type='median',
            pipeline_schematic=None):
        """
        Perform nested k-fold cross-validation on the data using the user-
        provided pipelines.

        This method is used in stages, depending on the output:
        1)  Train and score the pipelines on the inner-loop folds of each
            outer-fold. Decide the winning pipeline based on the highest mean
            or median score. Alert the user if no winner can be chosen because
            they have the same score. If a winner is chosen, go to step 3.
        2)  If no winning pipeline is chosen in a particular inner-loop contest,
            The rule dictated by the tie_breaker keyword argument will decide
            the winner. If tie_breaker is 'choice', the user is alerted and
            asked to run the fit method again with
            best_inner_fold_pipeline_inds keyword argument to decide the winner
            (preferably the simplest model).
        3)  If a winner is chosen for each inner-loop contest for each outer-
            loop in step 1 or the user designates one in step 2, the winning
            pipelines of all outer-folds are collected and the ultimate winner
            is chosen with the highest number of outer-folds it has won. If
            no winner is chosen, the user is alerted and asked to run the fit
            method again with the best_outer_fold_pipeline keyword argument to
            decide the final winner (again, preferably the simplest).
        4)  The ultimate winning pipeline is trained on the training set of the
            outer-folds and tested on it's testing set (the validation set),
            scored, and those values are used as an estimate of out-of-sample
            scoring. The final pipeline is then trained on all available data
            for use in prediction/production. A final report is output with
            details of the entire procedure.

        Parameters
        ----------
        X : numpy.ndarray, shape (m, n)
            Feature input matrix. Rows are values of each column feature for a
            given observation.

        y : numpy.ndarray, shape (m, )
            Target vector. Each entry is the output given the corresponding row
            of feature inputs in X.

        pipelines : list of sklearn.pipeline.Pipeline objects
            The scikit-learn pipelines that will be copied and evaluated

        stratified :    boolean, optional
            Determines if the data will be stratified so that the target labels
            in the resulting feature matrix and target vector will have the
            same overall composition as all of the data. This is a best
            practice for classification problems.
            True :  Stratify the data
            False : Don't stratify the data

        scoring_metric :    str, {'auc', 'rmse', 'accuracy'}, optional
            Metric used to score estimator.
            auc :   Area under the receiver operating characteristic (ROC)
                    curve.
            accuracy :  Percent of correctly classified targets
            rmse :  Root mean-squared error. Essentially, distance of actual
                    from predicted target values.

        tie_breaker :   str, {'choice', 'first'}, optional
            Decision rule to use to decide the winner in the event of a tie
            choice :    Inform the user that a tie has occured between
                        pipelines, either in the inner-loop contest or
                        outer-loop contest of the nested k-fold cross-
                        validation, and that they need to include either the
                        best_inner_fold_pipeline_inds or
                        best_outer_fold_pipeline keyword arguments when running
                        the fit method again to decide the winner(s).
            first :     Simply use the first pipeline, in the order provided,
                        with the same score.

        best_inner_fold_pipeline_inds : dict, optional
        best_outer_fold_pipeline
        score_type
        """

        # Build list of scikit-learn pipelines if not provided by user
        if pipelines is None:
            if pipeline_schematic is not None:
                # Form scikit-learn pipelines using the PipelineBuilder
                pipelines = PipelineBuilder().build_pipeline_bundle(
                                                            pipeline_schematic)
            else:
                raise Exception("A pipeline schematic keyword argument, " \
                    "pipeline_schematic, must be provided if no list of " \
                    "pipelines in the pipelines keyword argument is provided")

        if not best_outer_fold_pipeline:
            ######## Choose best inner fold pipelines for outer folds ########
            if not best_inner_fold_pipeline_inds:
                ############### Save inputs ###############
                self.X = X
                self.y = y

                self.scoring_metric = scoring_metric
                self.score_type = score_type

                ############### Check inputs ###############
                self.check_feature_target_data_consistent(self.X, self.y)

                # TODO: add check for pipelines once this is working

                ############ Shuffle data and save it and indices ############
                self.shuffle_data()

                ############### Save pipelines ###############
                if self.random_combinations is None:
                    self.pipelines = {pipeline_ind: pipeline \
                        for pipeline_ind, pipeline in enumerate(pipelines)}
                else:
                    shuffled_pipeline_inds = np.arange(len(pipelines))

                    random.seed(self.random_combination_seed)

                    random.shuffle(shuffled_pipeline_inds)

                    self.pipelines = {
                        int(pipeline_ind): pipelines[pipeline_ind] for pipeline_ind \
                        in shuffled_pipeline_inds[:self.random_combinations]
                    }

                ########## Derive outer and inner loop split indices ##########
                self.get_outer_split_indices(self.shuffled_X, y=self.shuffled_y,
                                             stratified=stratified)

                ########### Perform nested k-fold cross-validation ###########
                for outer_fold_ind, outer_fold in self.outer_folds.items():
                    outer_fold.fit(self.shuffled_X, self.shuffled_y,
                                   self.pipelines,
                                   scoring_metric=self.scoring_metric)
            else:
                for outer_fold_ind, best_pipeline_ind in \
                    best_inner_fold_pipeline_inds.items():

                    self.outer_folds[outer_fold_ind].fit(
                        self.shuffled_X, self.shuffled_y, self.pipelines,
                        scoring_metric=self.scoring_metric,
                        best_inner_fold_pipeline_ind=best_pipeline_ind)

        ############### Choose best outer fold pipeline ###############
        self.choose_best_outer_fold_pipeline(
            tie_breaker=tie_breaker,
            best_outer_fold_pipeline=best_outer_fold_pipeline)

        ############### Train winning pipeline on outer folds ###############
        self.train_winning_pipeline_on_outer_folds()

        ############### Train production pipeline ###############
        self.train_production_pipeline()

        ############### Output report ###############
        self.print_report()

    def train_production_pipeline(self):
        """
        Collects validation scores for and trains winning pipeline on all of
        the data for use in production
        """
        best_pipeline_ind = self.best_pipeline_ind

        if best_pipeline_ind is not None:
            ############### Initialize production pipeline ###############
            pipeline_kwargs = {
            'pipeline_id': best_pipeline_ind,
            'pipeline': clone(self.pipelines[best_pipeline_ind], safe=True),
            'scoring_metric': self.scoring_metric,
            'score_type': self.score_type,
            }

            self.pipeline = OuterFoldTrainedPipeline(**pipeline_kwargs)

            ############### Collect outer fold validation scores ###############
            inner_loop_train_scores = []
            inner_loop_test_scores = []
            for outer_fold in self.outer_folds.values():
                best_pipeline = outer_fold.pipelines[best_pipeline_ind]

                inner_loop_test_scores.append(best_pipeline.test_scores[0])
                inner_loop_train_scores.append(best_pipeline.train_scores[0])

            ############### Set scores in production pipeine ###############
            self.pipeline.set_inner_loop_scores(inner_loop_train_scores,
                                                inner_loop_test_scores)

            ############# Train production pipeline on all data #############
            # Accessing internal pipeline because the TrainedPipeline fit
            # method actually fits the data and then scores the resulting
            # pipeline. There is no test data when training on all data. Hence
            # it doesn't make sense to the pipeline after fitting.
            self.pipeline.pipeline.fit(self.shuffled_X, self.shuffled_y)

    def predict(self, X):
        """
        Uses the best pipeline to make a class prediction.
        """
        return self.pipeline.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Uses the best pipeline to give the probability of each result
        """
        return self.pipeline.pipeline.predict_proba(X)

    def train_winning_pipeline_on_outer_folds(self):
        """
        Trains and obtains validation scores for the winning model of the
        nested k-fold cross-validation inner loop contest if a winner has
        been chosen (self.best_pipeline_ind is set)
        """
        if self.best_pipeline_ind is not None:
            for outer_fold in self.outer_folds.values():
                outer_fold.train_winning_pipeline(self.best_pipeline_ind)

    def choose_best_outer_fold_pipeline(self, tie_breaker='choice',
                                        best_outer_fold_pipeline=None):
        """
        Pools winners of outer fold contests and selects ultimate winner by
        majority vote or by user choice (preferably simplest model for better
        out-of-sample performance) or some other decision rule,

        Parameters
        ----------
        tie_breaker :   str, {'choice', 'first'}
            Decision rule to use to decide the winner in the event of a tie
            choice :    Inform the user that they need to use the
                        choose_best_pipelines method to pick the winner of the
                        inner loop contest
            first :     Simply use the first model with the same score

        """
        # Collect all winning pipelines from each inner loop contest of each
        # outer fold
        outer_fold_winners = [outer_fold.best_pipeline_ind \
            for outer_fold_ind, outer_fold in self.outer_folds.items()]

        # Check if all folds have winners yet
        none_flag = False
        for outer_fold_winner in outer_fold_winners:
            if outer_fold_winner is None:
                none_flag = True


        if not none_flag:
            # Determine winner of all folds by majority vote
            counts = {x: outer_fold_winners.count(x) for x in outer_fold_winners}

            max_count = max([count for x, count in counts.items()])

            mode_inds = [x for x, count in counts.items() if count==max_count]

            best_pipeline_ind = None

            if best_outer_fold_pipeline is None:
                if len(mode_inds) == 1:
                    # Save winner if only one
                    best_pipeline_ind = mode_inds[0]
                else:
                    if tie_breaker=='choice':
                        # Encourage user to choose simplest model if there is no clear
                        # winner
                        for mode_ind in mode_inds:
                            print(mode_ind, self.pipelines[mode_ind])
                        print("\n\nNo model was chosen because there is no clear winner. " \
                              "Please use the same fit method with one of the "\
                              "indices above.\n\nExample:\tkfcv.fit(X.values, " \
                              "y.values, pipelines)\n\t\t"\
                              "kfcv.fit(X.values, y.values, pipelines, " \
                              "best_outer_fold_pipeline=9)")
                    elif tie_breaker=='first':
                        best_pipeline_ind = mode_inds[0]
            else:
                best_pipeline_ind = best_outer_fold_pipeline

            if best_pipeline_ind is not None:
                self.best_pipeline_ind = best_pipeline_ind

    def shuffle_data(self):
        """
        Shuffles and saves the feature data matrix, self.X, and target vector,
        self.y, if the self.shuffle_flag field is set to True and saves the
        corresponding indices as well.
        """
        # Calculate and save shuffled data indices
        self.get_shuffled_data_inds()

        # Shuffle and save data
        self.shuffled_X = self.X[self.shuffled_data_inds]
        self.shuffled_y = self.y[self.shuffled_data_inds]


    def get_shuffled_data_inds(self):
        """
        Calculates and saves shuffled data indices
        """
        point_count = self.X.shape[0]

        shuffled_data_inds = np.arange(point_count)

        if self.shuffle_flag:
            random.seed(self.shuffle_seed)

            random.shuffle(shuffled_data_inds)

        self.shuffled_data_inds = shuffled_data_inds

    def get_outer_split_indices(self, X, y=None, stratified=True):
        """
        Returns test-fold indices given the feature matrix, X, optional target
        values, y, and whether the split is to be stratified, stratified.
        """
        ################ Check inputs ###############
        self.check_feature_target_data_consistent(X, y)

        assert type(stratified) is bool, "The keyword argument determining " \
            "whether the splits are to be stratified or not, stratified, must" \
            " be boolean (True or False)."

        if stratified:
            assert y.any(), "Target value vector keyword argument, y, must " \
                "be present if stratified split keyword argument, stratified," \
                " is True."

        ################ Choose K-fold cross-validation type ################
        if not stratified or self.scoring_metric=='rmse':
            outer_k_fold_splitter = KFold(n_splits=self.outer_loop_fold_count,
                                    random_state=self.outer_loop_split_seed)
            outer_split_kwargs = {}

            inner_k_fold_splitters = \
                [KFold(n_splits=self.inner_loop_fold_count, random_state=seed) \
                for seed in self.inner_loop_split_seeds]
        else:
            outer_k_fold_splitter = StratifiedKFold(
                                n_splits=self.outer_loop_fold_count,
                                random_state=self.outer_loop_split_seed)

            outer_split_kwargs = {'y': y}

            inner_k_fold_splitters = \
                [StratifiedKFold(n_splits=self.inner_loop_fold_count,
                                 random_state=seed) \
                for seed in self.inner_loop_split_seeds]

        ################ Calculate and save outer and inner fold split indices ################
        for fold_id, (outer_train_inds, outer_test_inds) in enumerate(outer_k_fold_splitter.split(X,**outer_split_kwargs)):
            self.outer_folds[fold_id] = OuterFold(
                                            fold_id=fold_id,
                                            test_fold_inds=outer_test_inds,
                                            train_fold_inds=outer_train_inds)

            # Make sure the targets are available for a stratified run
            if not stratified:
                inner_split_kwargs = {}
            else:
                inner_split_kwargs = {'y': y[outer_train_inds]}

            # Get inner fold splitter for current outer fold
            inner_k_fold_splitter = inner_k_fold_splitters[fold_id]

            # Save inner fold test/train split indices
            for inner_fold_id, (inner_train_inds, inner_test_inds) in enumerate(inner_k_fold_splitter.split(X[outer_train_inds],**inner_split_kwargs)):
                self.outer_folds[fold_id].inner_folds[inner_fold_id] = \
                    Fold(
                        fold_id=inner_fold_id,
                        test_fold_inds=inner_test_inds,
                        train_fold_inds=inner_train_inds)

    def check_feature_target_data_consistent(self, X, y):
        """
        Checks to make sure the feature matrix and target vector are of the
        proper types and have the correct sizes
        """
        assert type(X) is np.ndarray, "Feature matrix, X, must be of type " \
            "numpy.ndarray."

        if y.any():
            assert type(y) is np.ndarray, "Target vector, y, must be of type " \
                "numpy.ndarray if given."

            assert len(y.shape) == 1, "Target vector must have a flat shape. " \
                "In other words the shape should be (m,) instead of (m,1) or " \
                "(1,m)."

        assert len(X.shape) == 2, "Feature matrix, X, must be 2-dimensional. " \
            "If the intention was to have only one data point with a single " \
            "value for each feature make the array (1,n). If there is only " \
            "one feature make the array nx1 (instead of just having a shape " \
            "of (n,))."

        if y.any():
            assert X.shape[0] == y.shape[0], "The number of rows of the " \
                "feature matrix, X, must match the length of the target " \
                "value vector, y, if given."

    def print_report(self):
        if self.best_pipeline_ind is not None:
            print(self.get_report())

    def get_report(self):
        """
        Generates report string
        """
        ############### Get validation scores for best pipeline ###############
        inner_loop_test_scores = self.pipeline.inner_loop_test_scores

        ############### Form pipeline string ###############
        pipeline_str = '\n'.join(['{}:\n{}\n'.format(*step) \
                                  for step in self.pipeline.pipeline.steps])

        ############### Build inner/outer-fold scores matrix ###############
        score_matrix = np.zeros([self.outer_loop_fold_count,
                                 self.inner_loop_fold_count])

        outer_fold_inds = []
        inner_fold_inds = []

        # Collect all outer- and inner-fold labels and populate score matrix
        for outer_fold_ind, outer_fold in self.outer_folds.items():
            if outer_fold_ind not in outer_fold_inds:
                outer_fold_inds.append(outer_fold_ind)
            for inner_fold_ind, inner_fold in outer_fold.inner_folds.items():
                if inner_fold_ind not in inner_fold_inds:
                    inner_fold_inds.append(inner_fold_ind)
                score = inner_fold.pipelines[ \
                                        self.best_pipeline_ind].test_scores[0]

                score_matrix[outer_fold_ind, inner_fold_ind] = score

        # Form headers for validation section
        quartile_headers = ['min', '25%', '50%', '75%', 'max']
        mean_based_headers = ['mean', 'std']
        outer_fold_headers = ['%d'%(outer_fold_ind) \
                              for outer_fold_ind in outer_fold_inds]

        # Get validation scores and their mean, std, and quartiles
        validation_scores = inner_loop_test_scores
        validation_mean = np.mean(inner_loop_test_scores)
        validation_std = np.std(inner_loop_test_scores, ddof=1)

        validation_quartiles = np.percentile(inner_loop_test_scores,
                                                [0, 25, 50, 75, 100])

        # Calculate means, standard deviations, and quartiles
        means = np.mean(score_matrix, axis=1)
        stds = np.std(score_matrix, axis=1, ddof=1)

        quartiles = np.percentile(score_matrix, [0, 25, 50, 75, 100],
                                    axis=1)

        # Initialize data report
        data_report = []

        # Form base header and data row format strings
        header_str = '{0:>4}{1:>10}{2:>15}'
        data_row_str = '{0:>4}{1:>10.4}{2:>15}'

        # Form in-report dividers based on number of outer folds
        data_report_divider = '----------------------  ------'

        data_report_divider += (10*len(outer_fold_inds))*'-'

        # Add additional columns based on number of outer folds
        inner_loop_contest_headers = ['','','']
        for outer_fold_ind_ind, outer_fold_ind in enumerate(outer_fold_inds):
            inner_loop_contest_headers.append('OF%d'%(outer_fold_ind))

            header_str += '{%d:>10}'%(outer_fold_ind_ind+3)

            data_row_str += '{%d:>10.4}'%(outer_fold_ind_ind+3)

        # Add quartile data rows
        data_report.append(header_str.format(*inner_loop_contest_headers))

        for quartile_header_ind, quartile_header in enumerate(quartile_headers):
            row_values = [quartile_header, validation_quartiles[quartile_header_ind], quartile_header]

            for outer_fold_quartile_score in quartiles[quartile_header_ind]:
                row_values.append(outer_fold_quartile_score)
            data_report.append(data_row_str.format(*row_values))

        data_report.append(data_report_divider)

        # Start mean data rows
        row_values = ['mean', validation_mean, 'mean']

        for outer_fold_mean_score in means:
            row_values.append(outer_fold_mean_score)

        data_report.append(data_row_str.format(*row_values))

        row_values = ['std', validation_std, 'std']
        for outer_fold_score_std in stds:
            row_values.append(outer_fold_score_std)

        data_report.append(data_row_str.format(*row_values))

        data_report.append(data_report_divider)

        # Fill rows where there are both validation and inner fold scores
        outer_fold_count = len(outer_fold_inds)

        inner_fold_count = len(inner_fold_inds)

        outer_fold_ind = 0
        inner_fold_ind = 0
        while outer_fold_ind <= outer_fold_count-1 \
            and inner_fold_ind <= inner_fold_count-1:

            row_values = ['OF%d'%(outer_fold_ind),
                          validation_scores[outer_fold_ind],
                          'IF%d'%(inner_fold_ind)]

            for outer_inner_fold_score in score_matrix.T[inner_fold_ind]:
                row_values.append(outer_inner_fold_score)

            data_report.append(data_row_str.format(*row_values))

            inner_fold_ind += 1
            outer_fold_ind += 1

        if outer_fold_ind <= outer_fold_count-1: # Still more outer folds
            while outer_fold_ind <= outer_fold_count-1:
                row_values = ['OF%d'%(outer_fold_ind),
                              validation_scores[outer_fold_ind],
                              '-']

                for outer_inner_fold_score in score_matrix.T[inner_fold_ind-1]:
                    row_values.append('-')

                data_report.append(data_row_str.format(*row_values))

                outer_fold_ind += 1
        elif inner_fold_ind <= inner_fold_count-1: # Still more innerfolds
            while inner_fold_ind <= inner_fold_count-1:

                row_values = ['-',
                              '-',
                              'IF%d'%(inner_fold_ind)]

                for outer_inner_fold_score in score_matrix.T[inner_fold_ind]:
                    row_values.append(outer_inner_fold_score)

                data_report.append(data_row_str.format(*row_values))

                inner_fold_ind += 1


        ############### Form and print report ###############
        str_inputs = {
            'data_report': '\n'.join(data_report),
            'data_report_divider': data_report_divider,
            'divider': 80*'-',
            'best_pipeline_ind': self.best_pipeline_ind,
            'pipeline': pipeline_str,
            'outer_loop_fold_count': self.outer_loop_fold_count,
            'inner_loop_fold_count': self.inner_loop_fold_count,
            'shuffle_seed': self.shuffle_seed,
            'outer_loop_split_seed': self.outer_loop_split_seed,
            'inner_loop_split_seeds': ', '.join(['%d'%(seed) \
                                     for seed in self.inner_loop_split_seeds]),
            'scoring_metric': self.scoring_metric,
            'score_type': self.score_type,

            'random_combinations': self.random_combinations,
            'random_combination_seed': self.random_combination_seed
        }

        report_str = \
        """
        {divider}
        Best pipeline: {best_pipeline_ind}
        {divider}
        {data_report_divider}
        Validation performance  Inner-loop scores
        {data_report_divider}
        {data_report}
        {data_report_divider}
        {divider}
        Pipeline steps
        ---------------
        {pipeline}
        {divider}
        Nested k-fold cross-validation parameters
        -----------------------------------------
        scoring metric:\t\t\t{scoring_metric}

        scoring type:\t\t\t{score_type}

        outer-fold count:\t\t{outer_loop_fold_count}
        inner-fold count:\t\t{inner_loop_fold_count}

        shuffle seed:\t\t\t{shuffle_seed}
        outer-loop split seed:\t\t{outer_loop_split_seed}
        inner-loop split seeds:\t\t{inner_loop_split_seeds}

        random combinations:\t\t{random_combinations}
        random combination seed:\t{random_combination_seed}
        {divider}

        """.format(**str_inputs)

        # Replace extra spaces resulting from indentation
        report_str = re.sub('\n        ', '\n', report_str)

        return report_str

    def plot_best_pipeline_scores(self, fontsize=10, figsize=(9, 3),
                                  markersize=8, draw_points=False,
                                  box_line_thickness=1):
        # Get data
        best_pipeline_data = {}
        for outer_fold_ind, outer_fold in self.outer_folds.items():
            best_pipeline_data[outer_fold_ind] = \
                outer_fold.pipelines[self.best_pipeline_ind].inner_loop_test_scores

        best_pipeline_data['val'] = self.pipeline.inner_loop_test_scores

        df = pd.DataFrame(best_pipeline_data)

        self.box_plot(df, x_label=self.scoring_metric, fontsize=fontsize,
                      figsize=figsize, markersize=markersize,
                      draw_points=draw_points,
                      box_line_thickness=box_line_thickness)

    def plot_contest(self, fontsize=6, figsize=(10, 30), markersize=2,
                     all_folds=False, color_by=None, color_map='viridis',
                     legend_loc='best', legend_font_size='10',
                     legend_marker_size=0.85, box_line_thickness=0.5,
                     draw_points=False, highlight_best=False):

        colors = None

        custom_legend = None

        # Collect pipeline data for each outer-fold contest
        pipeline_data = {pipeline_ind: {} for pipeline_ind in self.pipelines}

        for outer_fold_ind, outer_fold in self.outer_folds.items():
            for pipeline_ind, pipeline in outer_fold.pipelines.items():
                pipeline_data[pipeline_ind][outer_fold_ind] = \
                    outer_fold.pipelines[pipeline_ind].inner_loop_test_scores

        # Plot
        if not all_folds:
            # Do a separate box-and-whisker plot for each outer fold contest
            for outer_fold_ind in self.outer_folds:
                # Collect data for all pipelines corresponding to the current
                # outer-fold
                current_fold_data = {}
                for pipeline_ind in self.pipelines:
                    current_fold_data[pipeline_ind] = \
                        pipeline_data[pipeline_ind][outer_fold_ind]

                df = pd.DataFrame(current_fold_data)

                medians = df.median()

                medians.sort_values(ascending=True, inplace=True)

                df = df[medians.index]

                if color_by:
                    colors = self.get_colors(
                                df, color_by=color_by, color_map=color_map,
                                highlight_best=highlight_best)

                    custom_legend = self.get_custom_legend(
                                        df,
                                        color_by=color_by,
                                        color_map=color_map)

                self.box_plot(df, x_label=self.scoring_metric,
                              fontsize=fontsize, figsize=figsize,
                              markersize=markersize, colors=colors,
                              custom_legend=custom_legend,
                              legend_loc=legend_loc,
                              legend_font_size=legend_font_size,
                              legend_marker_size=legend_marker_size,
                              box_line_thickness=box_line_thickness,
                              draw_points=draw_points)

        else:
            # Combine all data for each pipeline and graph all together
            all_fold_data = {}
            for pipeline_ind, outer_fold_pipeline_data in pipeline_data.items():
                if pipeline_ind not in all_fold_data:
                    all_fold_data[pipeline_ind] = []

                for outer_fold_ind, outer_fold in self.outer_folds.items():
                    all_fold_data[pipeline_ind].extend(
                        pipeline_data[pipeline_ind][outer_fold_ind])

            df = pd.DataFrame(all_fold_data)

            medians = df.median()

            medians.sort_values(ascending=True, inplace=True)

            df = df[medians.index]

            if color_by:
                colors = self.get_colors(df, color_by=color_by,
                                         color_map=color_map,
                                         highlight_best=highlight_best)

                custom_legend = self.get_custom_legend(df,
                                                       color_by=color_by,
                                                       color_map=color_map)

            self.box_plot(df, x_label=self.scoring_metric,
                          fontsize=fontsize, figsize=figsize,
                          markersize=markersize, colors=colors,
                          custom_legend=custom_legend, legend_loc=legend_loc,
                          legend_font_size=legend_font_size,
                          legend_marker_size=legend_marker_size,
                          box_line_thickness=box_line_thickness,
                          draw_points=draw_points)

    def box_plot(self, df, x_label=None, fontsize=25, figsize=(15, 10),
                 markersize=12, colors=None, custom_legend=None,
                 legend_loc='best', legend_font_size='10',
                 legend_marker_size=0.85, box_line_thickness=1.75,
                 draw_points=False):
        """
        Plots all data in a dataframe as a box-and-whisker plot with optional
        axis label
        """
        tick_labels = [str(column) for column in df.columns]

        fontsize = fontsize

        # Draw figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Set background to opaque
        fig.patch.set_facecolor('white')

        # Set grid parameters
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle='--', which='both', color='black',
                       alpha=0.5, zorder=1)

        # Set left frame attributes
        ax.spines['left'].set_linewidth(1.8)
        ax.spines['left'].set_color('gray')
        ax.spines['left'].set_alpha(1.0)

        # Remove all but bottom frame line
        # ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Draw box plot
        box_plot_kwargs = dict(
            notch=0,
            sym='+',
            vert=False,
            whis=5,
            patch_artist=True,
            capprops=dict(
                color='k',
                linestyle='-',
                linewidth=box_line_thickness
            ),
            boxprops=dict(
                linestyle='-',
                linewidth=box_line_thickness,
                color='black'
            ),
            medianprops=dict(
                linestyle='none',
                color='k',
                linewidth=box_line_thickness
            ),
            whiskerprops=dict(
                color='k',
                linestyle='-',
                linewidth=box_line_thickness
            )

        )

        bp = plt.boxplot(df.values,**box_plot_kwargs)

        # Set custom colors
        if colors:
            for item in ['boxes']: #'medians' 'whiskers', 'fliers', 'caps'
                for patch, color in zip(bp[item],colors):
                    patch.set_color(color)

            for patch, color in zip(bp['medians'],colors):
                patch.set_color('black')
        else:
            for patch in bp['boxes']:
                patch.set_color('black')

            for patch in bp['medians']:
                patch.set_color('black')

        # Draw overlying data points
        if draw_points == True:
            for column_ind,column in enumerate(df.columns):
                # Get data
                y = (column_ind+1)*np.ones(len(df[column]))
                x = df[column].values

                # Plot data points
                plt.plot(x,y,'.',color='k',markersize=markersize)


        # Set tick labels and sizes
        plt.setp(ax, yticklabels=tick_labels)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), fontsize=fontsize)

        # Adjust limits so plot elements aren't cut off
        x_ticks, x_tick_labels = plt.xticks()

        # shift half of range to left
        range_factor = 2

        x_min = x_ticks[0]
        x_max = x_ticks[-1] + (x_ticks[-1] - x_ticks[-2])/float(range_factor)

        # Set new limits
        plt.xlim(x_min, x_max)

        # Set tick positions
        plt.xticks(x_ticks)

        # Place x- and y-labels
        plt.xlabel(x_label, size=fontsize)
        # plt.ylabel(y_label,size=small_text_size)

        # Move ticks to where I want them
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('left')

        if custom_legend:
            ax.legend(custom_legend[1], custom_legend[0],
                      handlelength=legend_marker_size,
                      handleheight=legend_marker_size,
                      frameon=False, loc=legend_loc)

            plt.setp(plt.gca().get_legend().get_texts(),
                     fontsize=legend_font_size)

        # Draw a white dot for medians
        for column_ind,column in enumerate(df.columns):
            x_median = np.median(df[column].values)
            y_median = (column_ind+1)*np.ones(1)

            # Plot data points
            plt.plot(x_median,y_median,'o',color='white',markersize=markersize,
                     markeredgecolor='white', zorder=3)

        # Display plot
        plt.show()

    def get_organized_pipelines(self, step_type=None):
        """
        Collects pipeline indices for each option (Ex: knn, svm,
        logistic_regression) for the desired step type (Ex: estimator).
        Collects pipeline indices of pipelines without the desired step type
        under a 'None' dictionary entry.

        Parameters
        ----------

        """
        organized_pipelines = {}

        if '__' in step_type:
            step_type, step_option, step_parameter = step_type.split('__')
        else:
            step_type, step_option, step_parameter = step_type, None, None

        for pipeline_ind, pipeline in self.pipelines.items():
            step_type_found = False

            # Does this pipeline have this step?
            for step in self.pipelines[pipeline_ind].steps:
                if step[0] == step_type:
                    step_type_found = True

                    step_name = step[1].__class__.__name__

                    # Are we interested in coloring by step parameter?
                    if step_option is not None and step_parameter is not None:
                    # if '__' in step_type:

                        parameter_name_found = False

                        step_parameters = step[1].get_params()

                        if step_parameter in step_parameters:
                            parameter_name_found = True

                            parameter_value = step_parameters[step_parameter]

                            value_parameter = "%s__%s"%(parameter_value,
                                                        step_parameter)

                            step_name = "%s__%s"%(value_parameter, step_name)

                    # Initialize the pipeline indices for this step name if not
                    # found
                    if step_name not in organized_pipelines:
                        organized_pipelines[step_name] = {
                            'pipeline_inds': []
                        }

                    organized_pipelines[step_name]['pipeline_inds'].append(
                                                                  pipeline_ind)

            # Lump pipeline in with default if step not found
            if not step_type_found:
                if 'None' not in organized_pipelines:
                    organized_pipelines['None'] = {
                        'pipeline_inds': []
                    }

                organized_pipelines['None']['pipeline_inds'].append(
                                                                  pipeline_ind)

        return organized_pipelines

    def order_by_parameter(self, parameter_str):
        parameter_value = parameter_str.split('__')[0]

        try:
            key_value = int(parameter_value)
        except:
            key_value = parameter_value

        return key_value

    def get_step_colors(self, df, color_by=None, color_map='viridis'):
        """
        """
        ######### Collect pipeline indices with desired attribute  #########
        step_colors = self.get_organized_pipelines(step_type=color_by)

        ############### Build working/indexible colormap ###############
        color_count = len(step_colors.keys())

        cmap = mpl_plt.get_cmap(color_map)

        cNorm = mpl_colors.Normalize(vmin=0, vmax=color_count-1)

        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        sorted_steps = sorted(step_colors.keys(), key=self.order_by_parameter)

        # Set internal colors
        color_ind = 0
        for step_name in sorted_steps:
            step = step_colors[step_name]

            if step_name == 'None':
                step['color'] = 'k'
            else:
                step['color'] = scalarMap.to_rgba(color_ind)

                color_ind += 1

        return step_colors

    def get_colors(self, df, color_by=None, color_map='viridis',
                   highlight_best=None):
        """
        """
        # Choose colors for each step option and collect corresponding
        # pipeline indices
        step_colors = self.get_step_colors(df, color_by=color_by,
                                           color_map=color_map)

        # Build colors list
        colors = []
        for pipeline_ind in df.columns:
            ind_found = False

            for step_name, step in step_colors.items():
                if pipeline_ind in step['pipeline_inds']:
                    colors.append(step['color'])

                    ind_found = True

            if not ind_found:
                colors.append(step_colors['None']['color'])

        return colors

    def get_custom_legend(self, df, color_by=None, color_map='viridis'):
        step_colors = self.get_step_colors(df, color_by=color_by,
                                           color_map=color_map)

        labels = sorted(step_colors.keys(), key=self.order_by_parameter)

        proxies = [self.create_proxy(step_colors[item]['color']) \
                   for item in labels]

        return (labels, proxies)

    def create_proxy(self, color):
        rect = plt.Rectangle((0,0), 1, 1, color=color)

        return rect

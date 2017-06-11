# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
import numpy as np
import pandas as pd
import random
import re

from sklearn.base import clone

# Graphing
import pylab as plt

# Cross validation tools
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .folds import Fold, OuterFold

from .trained_pipeline import OuterFoldTrainedPipeline

class NestedKFoldCrossValidation(object):
    """
    Class that handles nested k-fold cross validation, whereby the inner loop
    handles the model selection and the outer loop is used to provide an
    estimate of the chosen model's out of sample score.
    """
    def __init__(self, outer_loop_fold_count=3, inner_loop_fold_count=3,
                 outer_loop_split_seed=None, inner_loop_split_seeds=None,
                 shuffle_flag=True, shuffle_seed=None):
        ############### Initialize data ###############
        # Flag determining if initial data should be shuffled_y
        self.shuffle_flag = shuffle_flag

        # Seed determining shuffling of initial data
        self.shuffle_seed = shuffle_seed

        # Total number of folds in outer and inner loops
        self.outer_loop_fold_count = outer_loop_fold_count
        self.inner_loop_fold_count = inner_loop_fold_count

        # Seeds determining initial split of data into outer-folds
        self.outer_loop_split_seed = outer_loop_split_seed

        # List of seeds that will be used to split the training sets of the
        # the outer-folds into inner folds
        self.inner_loop_split_seeds = inner_loop_split_seeds

        # Shuffled data indices
        self.shuffled_data_inds = None

        # Test/train fold indices
        self.outer_folds = {}

        # Input data and targets
        self.X = None
        self.y = None

        # Shuffled data and targets
        self.shuffled_X = None
        self.shuffled_y = None

        # Pipelines
        self.pipelines = None

        # Best pipeline trained on all data
        self.pipeline = None

        # Winning pipeline for the entire process
        self.best_pipeline = {
            "best_pipeline_ind": None,
            "trained_all_pipeline": None,
            "validation_scores": [],
            "mean_validation_score": None,
            "validation_score_std": None,
            "median_validation_score": None,
            "interquartile_range": None,
            "confusion_matrix": None
        }

        self.score_type = None

        self.scoring_metric = None

        self.best_pipeline_ind = None

        ############### Populate fields with defaults ###############
        # Generate seed for initial shuffling of data if not provided
        if not self.shuffle_seed:
            self.shuffle_seed = random.randint(1,5000)

        # Generate seeds if not given (*200 since 5-fold CV results in range
        # of 1 to 1000)
        if not self.outer_loop_split_seed:
            self.outer_loop_split_seed = random.randint(
                                            1,
                                            self.outer_loop_fold_count*200)

        if not self.inner_loop_split_seeds:
            self.inner_loop_split_seeds = np.random.randint(
                                            1,
                                            high=self.inner_loop_fold_count*200,
                                            size=self.outer_loop_fold_count)

        ############### Check fields so far ###############
        outer_loop_fold_count_error = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer"

        assert type(self.outer_loop_fold_count) is int, outer_loop_fold_count_error

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

    def fit(self, X, y, pipelines, stratified=False, scoring_metric='auc',
            tie_breaker='choice', best_inner_fold_pipeline_inds=None,
            best_outer_fold_pipeline=None, score_type='median'):
        """
        Perform nested k-fold cross-validation on the data using the user-
        provided pipelines
        """
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
                self.pipelines = {pipeline_ind: pipeline \
                    for pipeline_ind, pipeline in enumerate(pipelines)}

                ########## Derive outer and inner loop split indices ##########
                self.get_outer_split_indices(self.shuffled_X, y=self.shuffled_y,
                                             stratified=stratified)

                ########### Perform nested k-fold cross-validation ###########
                for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
                    outer_fold.fit(self.shuffled_X, self.shuffled_y,
                                   self.pipelines,
                                   scoring_metric=self.scoring_metric)
            else:
                for outer_fold_ind, best_pipeline_ind in \
                    best_inner_fold_pipeline_inds.iteritems():

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
            for outer_fold_ind, outer_fold in self.outer_folds.iteritems()]

        # Check if all folds have winners yet
        none_flag = False
        for outer_fold_winner in outer_fold_winners:
            if outer_fold_winner is None:
                none_flag = True


        if not none_flag:
            # Determine winner of all folds by majority vote
            counts = {x: outer_fold_winners.count(x) for x in outer_fold_winners}

            max_count = max([count for x, count in counts.iteritems()])

            mode_inds = [x for x, count in counts.iteritems() if count==max_count]

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
                            print mode_ind, self.pipelines[mode_ind]
                        print "\n\nNo model was chosen because there is no clear winner. " \
                              "Please use the same fit method with one of the "\
                              "indices above.\n\nExample:\tkfcv.fit(X.values, " \
                              "y.values, pipelines)\n\t\tkfcv.train_winning_pipeline(3)"\
                              "kfcv.fit(X.values, y.values, pipelines, " \
                              "best_outer_fold_pipeline=9)"
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

    def get_outer_split_indices(self, X, y=None, stratified=False):
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
        if not stratified:
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
            print self.get_report()

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
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            if outer_fold_ind not in outer_fold_inds:
                outer_fold_inds.append(outer_fold_ind)
            for inner_fold_ind, inner_fold in outer_fold.inner_folds.iteritems():
                if inner_fold_ind not in inner_fold_inds:
                    inner_fold_inds.append(inner_fold_ind)
                score = inner_fold.pipelines[ \
                                        self.best_pipeline_ind].test_scores[0]

                score_matrix[outer_fold_ind, inner_fold_ind] = score

        # Form headers for validation section
        short_headers = ['0%', '25%', '50%', '75%', '100%', 'mean', 'std']

        quartile_headers = ['0%', '25%', '50%', '75%', '100%']
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
        {divider}

        """.format(**str_inputs)

        # Replace extra spaces resulting from indentation
        report_str = re.sub('\n        ', '\n', report_str)

        return report_str

    def plot_contest(self):
        # Get data
        best_pipeline_data = {}
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            best_pipeline_data[outer_fold_ind] = \
                outer_fold.pipelines[self.best_pipeline_ind].inner_loop_test_scores

        best_pipeline_data['val'] = self.pipeline.inner_loop_test_scores




        df = pd.DataFrame(best_pipeline_data)

        tick_labels = [str(column) for column in df.columns]

        number_size = 25


        # Draw figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))

        # Set background to opaque
        fig.patch.set_facecolor('white')

        # Set grid parameters
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle='--', which='both', color='black',
                       alpha=0.5)

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
        bp = plt.boxplot(df.values, notch=0, sym='+', vert=False, whis=5,
                         boxprops=dict(linestyle='-', linewidth=2,
                         color='black'),
                         medianprops=dict(linestyle='-',color='k',linewidth=2),
                         whiskerprops=dict(color='k',linewidth=2))

        # Draw overlying data points
        for column_ind,column in enumerate(df.columns):
            # Get data
            y = (column_ind+1)*np.ones(len(df[column]))
            x = df[column].values

            # Plot data points
            plt.plot(x,y,'.',color='k',markersize=12)

        # Set tick labels and sizes
        plt.setp(ax, yticklabels=tick_labels)
        plt.setp(ax.get_yticklabels(), fontsize=number_size)

        plt.setp(ax.get_xticklabels(), fontsize=number_size)

        # Place x- and y-labels
        plt.xlabel(self.scoring_metric,size=number_size)
        # plt.ylabel(y_label,size=small_text_size)

        # Display plot
        plt.show()

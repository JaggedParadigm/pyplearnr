# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
import numpy as np
import random
import re

from sklearn.base import clone

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

        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            print 'yay', outer_fold_ind, outer_fold.best_pipeline_ind
            print outer_fold.pipelines[self.best_pipeline_ind].test_scores

        print self.best_pipeline_ind


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

        print 'no ho', outer_fold_winners
        if not none_flag:
            # Determine winner of all folds by majority vote
            counts = {x: outer_fold_winners.count(x) for x in outer_fold_winners}

            max_count = max([count for x, count in counts.iteritems()])

            mode_inds = [x for x, count in counts.iteritems() if count==max_count]

            best_pipeline_ind = None

            print 'no ho', outer_fold_winners, counts, max_count, mode_inds, best_outer_fold_pipeline, best_outer_fold_pipeline is not None
            if best_outer_fold_pipeline is None:
                print '1'
                if len(mode_inds) == 1:
                    print '1-1'
                    # Save winner if only one
                    best_pipeline_ind = mode_inds[0]
                else:
                    print '1-2'
                    if tie_breaker=='choice':
                        print '1-2-1'
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
                        print '1-2-2'
                        best_pipeline_ind = mode_inds[0]
            else:
                print '2'
                best_pipeline_ind = best_outer_fold_pipeline

            if best_pipeline_ind is not None:
                print '3'

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
        inner_loop_test_scores = self.pipeline.inner_loop_test_scores
        inner_loop_train_scores = self.pipeline.inner_loop_train_scores

        ############### Form validation score string ###############
        validation_str_list = ['%1.5f'%(score) \
                               for score in inner_loop_test_scores]

        validation_score_str = ', '.join(validation_str_list)

        ############### Form validation score quartiles ###############
        q100, q75, q50, q25, q0 = np.percentile(inner_loop_test_scores,
                                                [0, 25, 50, 75, 100])

        iqr_str = '\t'.join(['%1.5f'%percentile \
                             for percentile in [q100, q75, q50, q25, q0]])

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

        # Calculate means, standard deviations, and quartiles
        means = np.mean(score_matrix, axis=1).reshape(-1,1)
        stds = np.std(score_matrix, axis=1, ddof=1).reshape(-1,1)

        percentiles = np.percentile(score_matrix, [0, 25, 50, 75, 100],
                                    axis=1).T

        # Concatenate float arrays and convert to strings
        float_matrix = np.concatenate((score_matrix, means, stds, percentiles),
                                      axis=1)

        str_matrix = np.zeros(float_matrix.shape)
        for row_ind, row in enumerate(float_matrix):
            for col_ind, entry in enumerate(row):
                str_matrix[row_ind, col_ind] = '%1.5f'%(entry)

        # Form header row array
        headers = np.array([''] + ['%d'%(inner_fold_ind) \
                                   for inner_fold_ind in inner_fold_inds] \
                                   + ['mean', 'std', '0%', '25%', '50%', '75%',
                                      '100%']).reshape(1,-1)

        # Form outer fold index array
        outer_fold_ind_str = np.array(['%d'%(outer_fold_ind) \
                          for outer_fold_ind in outer_fold_inds]).reshape(-1,1)

        # Concatenate to form final matrix
        final_matrix = np.concatenate((headers,
                                       np.concatenate((
                                            outer_fold_ind_str,
                                            str_matrix),axis=1)),axis=0)

        # Form matrix string
        matrix_str = []
        for row in final_matrix:
            matrix_str.append('\t'.join(row))

        # Form headers for validation section
        short_headers = ['0%', '25%', '50%', '75%', '100%', 'mean', 'std']

        ############### Form and print report ###############
        str_inputs = {
            'divider': 90*'-',
            'best_pipeline_ind': self.best_pipeline_ind,
            'pipeline': pipeline_str,
            'validation_scores': validation_score_str,
            'mean': '%1.5f'%(np.mean(inner_loop_test_scores)),
            'std': '%1.5f'%(np.std(inner_loop_test_scores, ddof=1)),
            'median': np.median(inner_loop_test_scores),
            'iqr': iqr_str,
            'test_matrix': '\n'.join(matrix_str),
            'outer_loop_fold_count': self.outer_loop_fold_count,
            'inner_loop_fold_count': self.inner_loop_fold_count,
            'shuffle_seed': self.shuffle_seed,
            'outer_loop_split_seed': self.outer_loop_split_seed,
            'inner_loop_split_seeds': ', '.join(['%d'%(seed) \
                                     for seed in self.inner_loop_split_seeds]),
            'scoring_metric': self.scoring_metric,
            'score_type': self.score_type,
            'short_headers': '\t'.join(short_headers),
            'tab_validation_scores': '\t'.join(validation_str_list)
        }

        report_str = \
        """
        {divider}
        Best pipeline:\t\t{best_pipeline_ind}
        {divider}

        {divider}
        Validation performance
        ----------------------
        {short_headers}
        {iqr}\t{mean}\t{std}

        {tab_validation_scores}

        {divider}
        Pipeline steps
        ---------------
        {pipeline}

        {divider}
        Inter-loop performance
        -----------------------
        {test_matrix}

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

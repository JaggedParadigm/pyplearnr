# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
import numpy as np
import random

from sklearn.base import clone

# Cross validation tools
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .folds import Fold, OuterFold

class NestedKFoldCrossValidation(object):
    """
    Class that handles nested k-fold cross validation, whereby the inner loop
    handles the model selection and the outer loop is used to provide an
    estimate of the chosen model's out of sample score.
    """
    def __init__(self, outer_loop_fold_count=3, inner_loop_fold_count=3,
                 outer_loop_split_seed=None, inner_loop_split_seed=None,
                 shuffle_flag=True, shuffle_seed=None):
        ############### Initialize data ###############
        # Flag determining if initial data should be shuffled_y
        self.shuffle_flag = shuffle_flag

        # Seed determining shuffling of initial data
        self.shuffle_seed = shuffle_seed

        # Total number of folds in outer and inner loops
        self.outer_loop_fold_count = outer_loop_fold_count
        self.inner_loop_fold_count = inner_loop_fold_count

        # Seeds determining splits
        self.outer_loop_split_seed = outer_loop_split_seed
        self.inner_loop_split_seed = inner_loop_split_seed

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

        self.scoring_metric = None

        ############### Populate fields with defaults ###############
        # Generate seed for initial shuffling of data if not provided
        if not self.shuffle_seed:
            self.shuffle_seed = random.randint(1,5000)

        # Generate seeds if not given (*20 since 5-fold CV results in range
        # of 1 to 100)
        if not self.outer_loop_split_seed:
            self.outer_loop_split_seed = random.randint(
                                            1,
                                            self.outer_loop_fold_count*20)

        if not self.inner_loop_split_seed:
            self.inner_loop_split_seed = random.randint(
                                            1,
                                            self.inner_loop_fold_count*20)

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

        assert type(self.inner_loop_split_seed) is int, "The " \
            "inner_loop_split_seed keyword argument, dictating how the data "\
            "is split into folds for the inner loop, must be an integer."

    def fit(self, X, y, pipelines, stratified=False, scoring_metric='auc'):
        """
        Perform nested k-fold cross-validation on the data using the user-
        provided pipelines
        """
        ############### Save inputs ###############
        self.X = X
        self.y = y

        self.scoring_metric = scoring_metric

        ############### Check inputs ###############
        self.check_feature_target_data_consistent(self.X, self.y)

        # TODO: add check for pipelines once this is working

        ############### Shuffle data and save ###############
        point_count = X.shape[0]

        # Get shuffle indices
        if self.shuffle_flag:
            self.shuffled_data_inds = self.get_shuffled_data_inds(point_count)
        else:
            # Make shuffle indices those which will result in same array if
            # shuffling isn't desired
            self.shuffled_data_inds = np.arange(point_count)

        # Shuffle data and save
        self.shuffled_X = X[self.shuffled_data_inds]
        self.shuffled_y = y[self.shuffled_data_inds]

        ############### Save pipelines ###############
        self.pipelines = {pipeline_ind: pipeline \
            for pipeline_ind, pipeline in enumerate(pipelines)}

        ############### Nested k-fold cross-validation ###############
        # Derive outer and inner loop split indices
        self.get_outer_split_indices(self.shuffled_X, y=self.shuffled_y,
                                     stratified=stratified)

        # Perform nested k-fold cross-validation
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            outer_fold.fit(self.shuffled_X, self.shuffled_y,
                           self.pipelines, scoring_metric=scoring_metric)

       ############### Pick and train winning pipeline ###############
        self.pick_train_winning_pipeline()

    def pick_train_winning_pipeline(self, tie_breaker='choice'):
        """
        Chooses winner of nested k-fold cross-validation as the majority vote
        of each outer fold winner and trains i
        """
        # Collect all winning pipelines from each inner loop contest of each
        # outer fold
        outer_fold_winners = [outer_fold.best_pipeline_ind \
            for outer_fold_ind, outer_fold in self.outer_folds.iteritems()]

        # Determine winner of all folds by majority vote
        counts = {x: outer_fold_winners.count(x) for x in outer_fold_winners}

        max_count = max([count for x, count in counts.iteritems()])

        mode_inds = [x for x, count in counts.iteritems() if count==max_count]

        if len(mode_inds) == 1:
            # Save winner if only one
            self.train_winning_pipeline(mode_inds[0])
        else:
            if tie_breaker=='choice':
                # Encourage user to choose simplest model if there is no clear
                # winner
                for mode_ind in mode_inds:
                    print mode_ind, self.pipelines[mode_ind]
                print "\n\nNo model was chosen because there is no clear winner. " \
                      "Please use the train_winning_pipeline method with one of the "\
                      "indices above.\n\nExample:\tkfcv.fit(X.values, " \
                      "y.values, pipelines)\n\t\tkfcv.train_winning_pipeline(3)"
            elif tie_breaker=='first':
                # Just set index to first mode
                self.train_winning_pipeline(mode_inds[0])

    def train_winning_pipeline(self, winning_pipeline_ind):
        """
        Simply sets the index of the best pipeline and trains it on all of the
        training data.
        """
        self.best_pipeline_ind = winning_pipeline_ind

        self.best_pipeline['best_pipeline_ind'] = self.best_pipeline_ind

        # Make sure the pipeline that won has its validation score set in each
        # outer fold
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            outer_fold.train_winning_pipeline(self.best_pipeline_ind,
                                              self.scoring_metric)


        # Collects scores for best pipeline
        validation_scores = self.best_pipeline['validation_scores']
        classification_reports = []
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            validation_scores.append(
                outer_fold.pipelines[
                    self.best_pipeline_ind]['validation_score']
                    )

            classification_reports.append(outer_fold.pipelines[
                            self.best_pipeline_ind]['test_classification_report']
                            )

        # Train best pipeline on all shuffled data for production
        self.best_pipeline['trained_all_pipeline'] = clone(
                                        self.pipelines[self.best_pipeline_ind],
                                        safe=True
                                        )

        self.best_pipeline['trained_all_pipeline'].fit(self.shuffled_X,
                                                       self.shuffled_y)

        # Output report detailing pipeline steps and statistics
        self.print_report()

    def print_report(self):
        print self.get_report()

    def get_report(self):
        """
        Generates report string
        """
        # print classification_reports

        print self.best_pipeline['validation_scores']

        self.best_pipeline['mean_validation_score'] = np.mean(validation_scores)
        self.best_pipeline['median_validation_score'] = np.median(
                                                        validation_scores
                                                        )
        self.best_pipeline['validation_score_std'] = np.std(validation_scores,
                                                            ddof=1)

        format_str = '{0:20}{1:20}{2:1}{3:<10}'

        blank_line = ['','','','']

        pipeline_str = []
        for step_ind,step in enumerate(
                                self.best_pipeline['trained_all_pipeline'].steps
                                ):
            step_name = step[0]

            step_obj = step[1]

            step_class = step_obj.__class__.__name__

            numbered_step = '%d: %s'%(step_ind+1, step_name)

            pipeline_str.append(format_str.format(*[numbered_step,
                                                    step_class,'','']))

            pipeline_str.append(format_str.format(*blank_line)) # Blank line

            step_fields = step_obj.get_params(deep=False)

            # Add fields to list of formatted strings
            step_fields = [format_str.format(*['',field,' = ',field_value]) \
                                 for field,field_value in step_fields.iteritems()]

            pipeline_str.append('\n'.join(step_fields))

            pipeline_str.append(format_str.format(*blank_line))

        pipeline_str = '\n'.join(pipeline_str)

        if self.scoring_metric == 'rmse':
            report = (
            "\nPipeline:\n\n%s\n"
            "\nTraining L2 norm score: %1.3f"
            "'\nTest L2 norm score: %1.3f"
            "\n\nGrid search parameters:\n\n%s\n"
            )%(pipeline_str,train_score,test_score)
        else:
            report = (
            "\nPipeline:\n\n%s\n"
            # "\nTraining set classification accuracy:\t%1.3f"
            "\nTest set classification accuracy:\t%1.3f"
            # "\n\nConfusion matrix:"
            # "\n\n%s"
            # "\n\nNormalized confusion matrix:"
            # "\n\n%s"
            # "\n\nClassification report:\n\n%s"
            # "\n\nGrid search parameters:\n\n%s\n"

            )%(pipeline_str,self.best_pipeline['median_validation_score'])#,
            #    np.array_str(confusion_matrix),np.array_str(normalized_confusion_matrix),
            #    classification_report)

        return report

    def get_shuffled_data_inds(self, point_count):
        """
        Returns shuffled data indices given number of data points
        """
        shuffled_data_inds = np.arange(point_count)

        np.random.shuffle(shuffled_data_inds)

        return shuffled_data_inds

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

            inner_k_fold_splitter = KFold(
                                        n_splits=self.inner_loop_fold_count,
                                        random_state=self.inner_loop_split_seed)

        else:
            outer_k_fold_splitter = StratifiedKFold(
                                n_splits=self.outer_loop_fold_count,
                                random_state=self.outer_loop_split_seed)

            outer_split_kwargs = {'y': y}

            inner_k_fold_splitter = StratifiedKFold(
                                        n_splits=self.inner_loop_fold_count,
                                        random_state=self.inner_loop_split_seed)

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

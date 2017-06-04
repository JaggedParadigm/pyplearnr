# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
import numpy as np
from scipy import stats
import random
import pandas as pd
import itertools

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer

# Feature selection tools
from sklearn.feature_selection import SelectKBest, f_classif

# Unsupervised learning tools
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

# Regression tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Cross validation tools
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.utils import shuffle

from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Classification metrics
import sklearn.metrics as sklearn_metrics

# Visualization
import matplotlib.pylab as plt

class NestedKFoldCrossValidation(object):
    """
    Class that handles nested k-fold cross validation, whereby the inner loop
    handles the model selection and the outer loop is used to provide an
    estimate of the chosen model's out of sample score.
    """
    def __init__(self, outer_loop_fold_count=3, inner_loop_fold_count=3,
                 outer_loop_split_seed=None, inner_loop_split_seed=None,
                 shuffle_flag=True, shuffle_seed=None):
        ############### Check input types ###############
        outer_loop_fold_count_error = "The outer_loop_fold_count" \
            " keyword argument, dictating the number of folds in the outer " \
            "loop, must be a positive integer"

        assert type(outer_loop_fold_count) is int, outer_loop_fold_count_error

        assert outer_loop_fold_count > 0, outer_loop_fold_count_error

        inner_loop_fold_count_error = "The inner_loop_fold_count" \
            " keyword argument, dictating the number of folds in the inner" \
            " loop, must be a positive integer"

        assert type(inner_loop_fold_count) is int, inner_loop_fold_count_error

        assert inner_loop_fold_count > 0, inner_loop_fold_count_error

        ############### Initialize fields ###############
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
            "mean_validation_score": None,
            "validation_score_std": None
        }

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

        ############### Check fields ###############
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
        ############### Check inputs ###############
        self.check_feature_target_data_consistent(X, y)

        # TODO: add check for pipelines once this is working

        self.X = X
        self.y = y

        self.pipelines = {pipeline_ind: pipeline for pipeline_ind, pipeline in enumerate(pipelines)}

        self.scoring_metric = scoring_metric

        point_count = X.shape[0]

        # Get shuffle indices
        if self.shuffle_flag:
            self.shuffled_data_inds = self.get_shuffled_data_inds(point_count)
        else:
            # Make shuffle indices those which will result in same array if
            # shuffling isnt desired
            self.shuffled_data_inds = np.arange(point_count)

        # Calculate outer and inner loop split indices
        self.get_outer_split_indices(X, y=y, stratified=stratified)

        # Shuffle data
        self.shuffled_X = X[self.shuffled_data_inds]
        self.shuffled_y = y[self.shuffled_data_inds]

        # Perform nested k-fold cross-validation
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            outer_fold.fit(self.shuffled_X, self.shuffled_y,
                           self.pipelines, scoring_metric=scoring_metric)

        self.pick_winning_pipeline()

    def train_winning_pipeline(self, winning_pipeline_ind):
        """
        Simply sets the index of the best pipeline and trains it on all of the
        training data.
        """
        # Make sure the pipeline that won has its validation score set in each
        # outer fold
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():
            winning_pipeline = outer_fold.pipelines[winning_pipeline_ind]

            validation_score = winning_pipeline['validation_score']

            if not validation_score:
                pass


            # outer_fold.pipelines[]

        """
        self.pipelines[pipeline_id] = {
            'id': pipeline_id,

            'pipeline': clone(pipeline, safe=True),

            'mean_inner_test_score': None,
            'median_inner_test_score': None,
            'inner_test_score_std': None,

            'mean_inner_train_score': None,
            'median_inner_train_score': None,
            'inner_train_score_std': None,

            'all_trained_pipeline': None,

            'y_validation_pred': None,
            'validation_score': None,

            'outer_y_train_pred': None,
            'outer_train_score': None,

            'scoring_metric': None
        }

        """

        # self.best_pipeline["best_pipeline_ind"] = winning_pipeline_ind
        #
        # # Clone winning pipeline
        # self.best_pipeline["trained_all_pipeline"] = clone(
        #     self.pipelines[winning_pipeline_ind], safe=True
        #     )
        #
        # # # Train winning pipeline
        # # for outer_fold in self.outer_folds:
        # #     outer_fold.
        #
        #
        # self.best_pipeline = {
        #     "best_pipeline_ind": None,
        #     "trained_all_pipeline": None,
        #     "mean_validation_score": None,
        #     "validation_score_std": None
        # }

    def pick_winning_pipeline(self, tie_breaker='choice'):
        """
        Chooses winner of nested k-fold cross-validation as the majority vote of
        each outer fold winner
        """
        # Collect all winning pipelines in for inner loop contest of each outer
        # fold
        outer_fold_winners = [outer_fold.best_pipeline_ind \
                              for outer_fold_ind, outer_fold in self.outer_folds.iteritems()]

        # Determine winner of all folds by majority vote
        counts = {x: outer_fold_winners.count(x) for x in outer_fold_winners}

        max_count = max([count for x, count in counts.iteritems()])

        mode_inds = [x for x, count in counts.iteritems() if count==max_count]

        if len(mode_inds) == 1:
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
            self.outer_folds[fold_id] = OuterFoldInds(
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
                    FoldInds(
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

class FoldInds(object):
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
                'y_test_pred': None,
                'y_train_pred': None,
                'test_score': None,
                'train_score': None,
                'pipeline': clone(pipeline, safe=True)
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

class OuterFoldInds(FoldInds):
    """
    Class containing test/train split indices for data
    """
    def __init__(self, fold_id=None, test_fold_inds=None,
                 train_fold_inds=None):
        ############### Initialize fields ###############
        super(OuterFoldInds, self).__init__(fold_id=fold_id,
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
        self.X_test = shuffled_X[self.test_fold_inds]
        self.y_test = shuffled_y[self.test_fold_inds]

        self.X_train = shuffled_X[self.train_fold_inds]
        self.y_train = shuffled_y[self.train_fold_inds]

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

                'mean_inner_test_score': None,
                'median_inner_test_score': None,
                'inner_test_score_std': None,

                'mean_inner_train_score': None,
                'median_inner_train_score': None,
                'inner_train_score_std': None,

                'all_trained_pipeline': None,

                'y_validation_pred': None,
                'validation_score': None,

                'outer_y_train_pred': None,
                'outer_train_score': None,

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

        # Calculate outer loop training score
        best_pipeline['outer_train_score'] = \
            PipelineEvaluator().get_score(self.y_train,
                                          best_pipeline['outer_y_train_pred'],
                                          scoring_metric)

        # Calculate validation score
        best_pipeline['validation_score'] = \
            PipelineEvaluator().get_score(self.y_test,
                                          best_pipeline['y_validation_pred'],
                                          scoring_metric)

class PipelineEvaluator(object):
    """
    Class used to evaluate pipelines
    """
    def get_score(self, y, y_pred, scoring_metric):
        """
        Returns the score given target values, predicted values, and a scoring
        metric.
        """
        if scoring_metric == 'auc':
            # Get ROC curve points
            false_positive_rate, true_positive_rate, _ = sklearn_metrics.roc_curve(y,
                                                                                   y_pred)

            # Calculate the area under the curve
            score =  sklearn_metrics.auc(false_positive_rate, true_positive_rate)

        return score

class PipelineBundle(object):
    """
    Collection of
    """
    def __init__(self):
        pass

    def build_pipeline_bundle(self, pipeline_bundle_schematic):
        """
        Returns a list of scikit-learn pipelines given a pipeline bundle
        schematic.

        TODO: Create a comprehensive description of the pipeline schematic

        The general form of the pipeline bundle schematic is:

        pipeline_bundle_schematic = [
            step_1,
            ...
            step_n
        ]

        Steps take the form:

        step_n = {
            'step_n_type': {
                'none': {}, # optional, used to not include the step as a permutation
                step_n_option_1: {},
            }


        }

        pipeline_bundle_schematic = [
            {'step_1_type': {
                'none': {}
                'step_1': {
                    'step_1_parameter_1': [step_1_parameter_1_value_1 ... step_1_parameter_1_value_p]
                    ...
                    'step_1_parameter_2': [step_1_parameter_2_value_1 ... step_1_parameter_2_value_m]
                }
            }},
            ...
        ]
        """
        # Get supported scikit-learn objects
        sklearn_packages = self.get_supported_sklearn_objects()

        # Obtain all corresponding scikit-learn package options with all
        # parameter combinations for each step
        pipeline_options = []
        for x in pipeline_bundle_schematic:
            step_name = x.keys()[0]

            step_options = x[step_name]

            step_iterations = []
            for step_option, step_parameters in step_options.iteritems():
                parameter_names = step_parameters.keys()

                parameter_combos = [step_parameters[step_parameter_name] \
                                    for step_parameter_name in parameter_names]

                for parameter_combo in list(itertools.product(*parameter_combos)):
                    parameter_kwargs = {pair[0]: pair[1] \
                                        for pair in zip(parameter_names,
                                                        parameter_combo)}

                    if step_option != 'none':
                        step = (step_name, sklearn_packages[step_name][step_option](**parameter_kwargs))

                        step_iterations.append(step)

            pipeline_options.append(step_iterations)

        # Form all step/parameter permutations and convert to scikit-learn pipelines
        pipelines = [Pipeline(x) for x in itertools.product(*pipeline_options)]

        return pipelines

    def get_supported_sklearn_objects(self):
        """
        Returns supported scikit-learn estimators, selectors, and transformers
        """
        sklearn_packages = {
            'feature_selection': {
                'select_k_best': SelectKBest
            },
            'scaler': {
                'standard': StandardScaler,
                'normal': Normalizer,
                'min_max': MinMaxScaler,
                'binary': Binarizer
            },
            'transform': {
                'pca': PCA
        #         't-sne': pipeline_TSNE(n_components=2, init='pca')
            },
            'estimator': {
                'knn': KNeighborsClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
        #         'polynomial_regression': PolynomialFeatures(), Need to find a different solution for polynomial regression
                'multilayer_perceptron': MLPClassifier,
                'random_forest': RandomForestClassifier,
                'adaboost': AdaBoostClassifier
            }

        }

        return sklearn_packages

    def get_default_pipeline_step_parameters(self,feature_count):
        # Set pre-processing pipeline step parameters
        pre_processing_grid_parameters = {
            'select_k_best': {
                'k': range(1,feature_count+1)
            }
        }

        # Set classifier pipeline step parameters
        classifier_grid_parameters = {
            'knn': {
                'n_neighbors': range(1,31),
                'weights': ['uniform','distance']
            },
            'logistic_regression': {
                'C': np.logspace(-10,10,5)
            },
            'svm': {},
            'multilayer_perceptron': {
                'hidden_layer_sizes': [[x] for x in range(min(3,feature_count),
                                                          max(3,feature_count)+1)]
            },
            'random_forest': {
                'n_estimators': range(90,100)
            },
            'adaboost': {}
        }

        # Set regression pipeline step parameters
        regression_grid_parameters = {
            'polynomial_regression': {
                'degree': range(1,5)
            }
        }

        # Return defaults
        return pre_processing_grid_parameters,classifier_grid_parameters,regression_grid_parameters

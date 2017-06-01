# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
import numpy as np
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

        self.pipelines = {}

    def fit(self, inner_loop_X_train, inner_loop_y_train, inner_loop_X_test,
            inner_loop_y_test, pipelines, scoring_metric='auc'):
        """
        Fits the pipeslines to the current inner fold training data
        """
        for pipeline_id, pipeline in pipelines.iteritems():
            print 2*'\t', pipeline_id

            # # Form id for this inner fold, outer fold, and pipeline
            # # combination
            # outer_inner_pipeline_id = "%d_%d_%d"%(outer_fold_ind,
            #                                       inner_fold_ind,
            #                                       pipeline_id)

            # Initialize this combination
            self.pipelines[pipeline_id] = {
                'id': pipeline_id,
                'test_score': None,
                'train_score': None,
                'pipeline': clone(pipeline, safe=True)
            }

            # Fit pipeline to training set of fold
            self.pipelines[pipeline_id]['pipeline'].fit(inner_loop_X_train,
                                                        inner_loop_y_train)

            # Calculate predicted targets from input test data
            inner_loop_y_test_pred = self.pipelines[pipeline_id]['pipeline'].predict(
                                        inner_loop_X_test)

            # Calculate the training prediction
            inner_loop_y_train_pred = self.pipelines[pipeline_id]['pipeline'].predict(
                                        inner_loop_X_train)

            # Calculate train score
            self.pipelines[pipeline_id]['train_score'] = PipelineEvaluator().get_score(
                                                        inner_loop_y_train,
                                                        inner_loop_y_train_pred,
                                                        scoring_metric)

            # Calculate test score
            self.pipelines[pipeline_id]['test_score'] = PipelineEvaluator().get_score(
                                                        inner_loop_y_test,
                                                        inner_loop_y_test_pred,
                                                        scoring_metric)

            print 3*'\t', self.pipelines[pipeline_id]['train_score'], self.pipelines[pipeline_id]['test_score']

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

        self.pipelines = {}

        self.best_pipeline = None

    def fit(self, outer_loop_X_train, outer_loop_y_train, pipelines,
            scoring_metric='auc'):
        """
        Performs inner loop of nested k-fold cross-validation for current outer
        fold
        """
        # Fit all pipelines to the training set of each inner fold
        for inner_fold_ind, inner_fold in self.inner_folds.iteritems():
            print '\t', inner_fold.fold_id

            current_inner_test_fold_inds = inner_fold.test_fold_inds
            current_inner_train_fold_inds = inner_fold.train_fold_inds

            inner_loop_X_test = outer_loop_X_train[current_inner_test_fold_inds]
            inner_loop_y_test = outer_loop_y_train[current_inner_test_fold_inds]

            inner_loop_X_train = outer_loop_X_train[current_inner_train_fold_inds]
            inner_loop_y_train = outer_loop_y_train[current_inner_train_fold_inds]

            inner_fold.fit(inner_loop_X_train, inner_loop_y_train,
                           inner_loop_X_test, inner_loop_y_test, pipelines,
                           scoring_metric=scoring_metric)

        # Calculate and save means and standard deviations for train/test fold
        # scores for each pipeline
        max_score = -1e-14
        max_ind = -1
        for pipeline_id, pipeline in pipelines.iteritems():
            # Initialize pipeline
            self.pipelines[pipeline_id] = {
                'id': pipeline_id,
                'pipeline': clone(pipeline, safe=True),
                'mean_test_score': None,
                'median_test_score': None,
                'test_score_std': None,
                'mean_train_score': None,
                'median_train_score': None,
                'train_score_std': None
            }

            # Collect test and train scores
            test_scores = []
            train_scores = []
            for inner_fold_ind in self.inner_folds:
                test_scores.append(self.inner_folds[inner_fold_ind].pipelines[pipeline_id]['test_score'])
                train_scores.append(self.inner_folds[inner_fold_ind].pipelines[pipeline_id]['train_score'])

            # Calculate and save statistics on test and train scores
            self.pipelines[pipeline_id]['mean_test_score'] = np.mean(test_scores)
            self.pipelines[pipeline_id]['median_test_score'] = np.median(test_scores)
            self.pipelines[pipeline_id]['test_score_std'] = np.std(test_scores,
                                                                   ddof=1)

            self.pipelines[pipeline_id]['mean_train_score'] = np.mean(train_scores)
            self.pipelines[pipeline_id]['median_train_score'] = np.median(train_scores)
            self.pipelines[pipeline_id]['train_score_std'] = np.std(train_scores,
                                                                    ddof=1)

            if max_score < self.pipelines[pipeline_id]['median_test_score']:
                max_score = self.pipelines[pipeline_id]['median_test_score']
                max_ind = pipeline_id

class NestedKFoldCrossValidation(object):
    """
    Class that handles nested k-fold cross validation, whereby the inner loop
    handles the model selection and the outer loop is used to provide an
    estimate of the chosen model's out of sample score.
    """
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
        shuffled_X = X[self.shuffled_data_inds]
        shuffled_y = y[self.shuffled_data_inds]

        # Perform nested k-fold cross-validation
        for outer_fold_ind, outer_fold in self.outer_folds.iteritems():


            print outer_fold.fold_id

            current_outer_test_fold_inds = outer_fold.test_fold_inds
            current_outer_train_fold_inds = outer_fold.train_fold_inds

            outer_loop_X_test = shuffled_X[current_outer_test_fold_inds]
            outer_loop_y_test = shuffled_y[current_outer_test_fold_inds]

            outer_loop_X_train = shuffled_X[current_outer_train_fold_inds]
            outer_loop_y_train = shuffled_y[current_outer_train_fold_inds]

            # Fit and score each pipeline on the inner folds of current
            # outer fold
            outer_fold.fit(outer_loop_X_train, outer_loop_y_train,
                           self.pipelines, scoring_metric='auc')


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

        self.pipelines = None

        # Combinations of an inner fold, an outer fold, and a pipeline
        self.outer_inner_pipeline_combos = {}

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





# Define custom TSNE class so that it will work with pipeline
class pipeline_TSNE(TSNE):
    def transform(self,X):
        return self.fit_transform(X)

class OptimizationBundle:
    """
    Collection of pipeline optimizations along with summary statistics and
    methods to compare them.
    """
    def __init__(self):
        # Initialize fields
        self.X = None # Feature input data
        self.y = None # Feature target data

        # Set only feature interaction option as None if not provided
        if not feature_interaction_options:
            feature_interaction_options = [None]
        else:
            self.feature_interaction_options = feature_interaction_options

        # Set feature selection options as None if not provided
        if not feature_selection_options:
            feature_selection_options = [None]

        # Set scaling options to None if not provided
        if not scaling_options:
            scaling_options = [None]

        # Set tranformation options to None if not provided
        if not transformations:
            transformations = [None]

        self.pipeline_optimizations = dict() # Pipeline optimizations

    def fit(self,X,y,estimators,
            feature_interaction_options=None,
            feature_selection_options=None,
            scaling_options=None,
            transformations=None,
            num_validation_repeats=1):
        """
        Performs repeated (stratified if classification) nested k-fold cross-validation
        to estimate out-of-sample classification accuracy
        """
        # Save data
        self.X = X.copy
        self.y = y.copy

        # Save number of validation repeats
        self.self.num_validation_repeats = self.num_validation_repeats

        # Perform model comparisons for each validation split
        for model_validation_comparison_ind in range(self.num_validation_repeats):
            # Split data
            pass

        pipeline_steps = [feature_interaction_options, feature_selection_options,
                          scaling_options, transformations, estimators]

        pipeline_options = list(itertools.product(*pipeline_steps))

        optimized_pipelines = {}

        pipeline_count = len(pipeline_options)

        for pipeline_step_combo_ind,pipeline_step_combo in enumerate(pipeline_options):
            model_name = []

            feature_interactions = pipeline_step_combo[0]

            model_name.append('interaction')

            if feature_interactions:
                model_name.append('interactions')

            feature_selection_type = pipeline_step_combo[1]

            if feature_selection_type:
                model_name.append('select')

            scale_type = pipeline_step_combo[2]

            if scale_type:
                model_name.append(scale_type)

            transform_type = pipeline_step_combo[3]

            if transform_type:
                model_name.append(transform_type)

            estimator = pipeline_step_combo[4]

            model_name.append(estimator)

            model_name = '_'.join(model_name)

            print(model_name,'%d/%d'%(pipeline_step_combo_ind,pipeline_count))

            # Set pipeline keyword arguments
            optimized_pipeline_kwargs = {
                'feature_selection_type': feature_selection_type,
                'scale_type': scale_type,
                'feature_interactions': feature_interactions,
                'transform_type': transform_type
                }

            # Initialize pipeline
            optimized_pipeline = ppl.OptimizedPipeline(
                estimator,
                **optimized_pipeline_kwargs
                )

            # Set pipeline fitting parameters
            fit_kwargs = {
                'cv': 10,
                'num_parameter_combos': None,
                'n_jobs': -1,
                'random_state': None,
                'suppress_output': True,
                'use_default_param_dist': True,
                'param_dist': None,
                'test_size': 0.2 # 20% saved as test set
            }

            # Fit data
            optimized_pipeline.fit(X,y,**fit_kwargs)

            # Save optimized pipeline
            optimized_pipelines[model_name] = optimized_pipeline

    def plot_test_scores(self):
        """
        """
        pass

    def get_options(self):
        """
        Returns all estimator, selection, scaling, transformation, and feature
        interaction options
        """
        supported_options = dict(
            estimators = ['knn','logistic_regression','svm',
                          'multilayer_perceptron','random_forest','adaboost'],
            feature_interaction_options = [False],
            feature_selection_options = [None,'select_k_best'],
            scaling_options = [None,'standard','normal','min_max','binary'],
            transformations = [None,'pca']
        )

        return supported_options


class PipelineOptimization:
    """
    Pipeline with optimal parameters found through nested k-folds cross
    validation
    """
    def __init__(self,estimator,
                 feature_selection_type=None,
                 scale_type=None,
                 transform_type=None):

        # Initialize fields
        self.X = None # feature inputs
        self.y = None # target values

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

        self.scale_type = scale_type # Type of scaling, if any

        self.transform_type = transform_type # Type of transformation, if any

        self.feature_selection_type=feature_selection_type # Type of feature selection

        self.estimator = estimator # Estimator
        self.estimator_type = None # Classification or regression

        self.param_dist=None # Parameters to be gridded over

        self.cv = None # k, in k-folds cross validation
        self.num_parameter_combos = None # Number of parameter combos to use (for RandomSearchCV)

        self.feature_count = None

        # Form pipeline
        self.pipeline,self.scoring = self.construct_pipeline(self.estimator,
                                                             feature_selection_type=self.feature_selection_type,
                                                             scale_type=self.scale_type,
                                                             transform_type=self.transform_type)

        # Initialize grid_search object field
        self.grid_search = None

        # Initialize user-provided pipeline parameters
        self.param_dist = None

        # Initialize optimized pipeline parameters
        self.score_type_ = None # "classification" or "regression" score
        self.train_score_ = None # Score from training set
        self.test_score_ = None # Score from test set

        self.confusion_matrix_ = None # Confusion matrix, if classification
        self.normalized_confusion_matrix_ = None # Confusion matrix divided by its total sum

        self.classification_report_ = None # Classification report

        self.best_parameters_ = None # Best parameters

    def fit(self,X,y,
            cv=10,
            num_parameter_combos=None,
            n_jobs=-1,
            random_state=None,
            suppress_output=False,
            use_default_param_dist=True,
            param_dist=None,
            test_size=0.2):
        """
        Uses the optimize_pipeline method to optimize object pipeline through nested k-folds cross validation
        """
        # Convert X and y to ndarray if either Pandas series or dataframe
        if type(X) is not np.ndarray:
            if type(X) is pd.core.frame.DataFrame or type(X) is pd.core.series.Series:
                X = np.array(X.values)
            else:
                raise Exception('Data input, X, must be of type pandas.core.frame.DataFrame, \
                                pandas.core.series.Series, or numpy.ndarray')

        if type(y) is not np.ndarray:
            if type(y) is pd.core.frame.DataFrame or type(y) is pd.core.series.Series:
                y = np.array(y.values)
            else:
                raise Exception('Data output, y, must be of type pandas.core.frame.DataFrame, \
                                pandas.core.series.Series, or numpy.ndarray')

        # Save data
        self.X = X.copy()
        self.y = y.copy()

        # Save cross-validation settings
        self.cv = cv
        self.num_parameter_combos = num_parameter_combos

        # Save number of features
        self.feature_count = self.X.shape[1]

        # Determine estimator type
        self.estimator_type = self.get_estimator_type(self.estimator,self.feature_count)

        # Make initial split stratified if classifier
        if self.estimator_type == 'classifier':
            stratify = self.y
        else:
            stratify = None

        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X,self.y,test_size=test_size,random_state=random_state,stratify=stratify)

        # Derive pipeline parameters to grid over
        self.param_dist = self.get_parameter_grid(self.estimator,self.feature_count,
                                                  feature_selection_type=self.feature_selection_type,
                                                  scale_type=self.scale_type,
                                                  param_dist=param_dist,
                                                  use_default_param_dist=use_default_param_dist)

        # Fit pipeline using nested k-folds cross validation
        self.grid_search = self.optimize_pipeline(self.X_train,self.y_train,self.pipeline,self.scoring,
                                                  self.param_dist,
                                                  cv=self.cv,
                                                  n_jobs=n_jobs,
                                                  num_parameter_combos=num_parameter_combos)

        # Save pipeline metrics and predicated values
        pipeline_metrics = self.test_pipeline(self.X_test,self.y_test,self.grid_search,self.estimator_type)

        self.y_pred = pipeline_metrics['y_pred']

        self.train_score_ = pipeline_metrics['train_score']
        self.test_score_ = pipeline_metrics['test_score']

        if self.estimator_type == 'classifier':
            self.confusion_matrix_ = pipeline_metrics['confusion_matrix']
            self.normalized_confusion_matrix_ = pipeline_metrics['normalized_confusion_matrix']
            self.classification_report_ = pipeline_metrics['classification_report']

        # Save best parameters
        self.best_parameters_ = self.grid_search.best_params_

        # Fit pipeline with best parameters obtained from grid search using all data
        self.pipeline.set_params(**self.grid_search.best_params_).fit(X,y)

        # Display model validation details
        if not suppress_output:
            print(self)

    def __str__(self):
        return self.get_report(self.pipeline,
                               self.grid_search,
                               self.feature_selection_type,
                               self.estimator,
                               self.estimator_type,
                               self.train_score_,
                               self.test_score_,
                               self.confusion_matrix_,
                               self.normalized_confusion_matrix_,
                               self.classification_report_)

    def get_report(self,pipeline,grid_search,feature_selection_type,
                   estimator,
                   estimator_type,
                   train_score,
                   test_score,
                   confusion_matrix,
                   normalized_confusion_matrix,
                   classification_report):

        # Collect pipeline steps and key,value pairs for all parameters
        format_str = '{0:20}{1:20}{2:1}{3:<10}'

        blank_line = ['','','','']

        pipeline_str = []
        for step_ind,step in enumerate(pipeline.steps):
            step_name = step[0]

            step_obj = step[1]

            step_class = step_obj.__class__.__name__

            numbered_step = '%d: %s'%(step_ind+1,step_name)

            pipeline_str.append(format_str.format(*[numbered_step,step_class,'','']))

            pipeline_str.append(format_str.format(*blank_line)) # Blank line

            step_fields = step_obj.get_params(deep=False)

            # Add fields to list of formatted strings
            step_fields = [format_str.format(*['',field,' = ',field_value]) \
                                 for field,field_value in step_fields.iteritems()]

            pipeline_str.append('\n'.join(step_fields))

            pipeline_str.append(format_str.format(*blank_line))

        pipeline_str = '\n'.join(pipeline_str)

        # Collect grid search key,value pairs for all parameters
        grid_search_fields = grid_search.get_params(deep=False)

        grid_search_str = [format_str.format(*['',field,' = ',field_value]) \
                           for field,field_value in grid_search_fields.iteritems()\
                            if field != 'estimator']

        grid_search_str.append(format_str.format(*blank_line))

        grid_search_str.append(format_str.format(*['Best parameters:','','','']))

        grid_search_str.append(format_str.format(*blank_line))

        grid_search_str.extend([format_str.format(*['',param_name,' = ',param_value]) \
                                for param_name,param_value in grid_search.best_params_.iteritems()])

        grid_search_str = '\n'.join(grid_search_str)


        if estimator_type == 'classifier':
            report = (
            "\nPipeline:\n\n%s\n"
            "\nTraining set classification accuracy:\t%1.3f"
            "\nTest set classification accuracy:\t%1.3f"
            "\n\nConfusion matrix:"
            "\n\n%s"
            "\n\nNormalized confusion matrix:"
            "\n\n%s"
            "\n\nClassification report:\n\n%s"
            "\n\nGrid search parameters:\n\n%s\n"

            )%(pipeline_str,train_score,test_score,
               np.array_str(confusion_matrix),np.array_str(normalized_confusion_matrix),
               classification_report,grid_search_str)
        elif estimator_type == 'regressor':
            report = (
            "\nPipeline:\n\n%s\n"
            "\nTraining L2 norm score: %1.3f"
            "'\nTest L2 norm score: %1.3f"
            "\n\nGrid search parameters:\n\n%s\n"
            )%(pipeline_str,train_score,test_score,grid_search_str)
        else:
            report = None

        return report

    def get_estimator_type(self,estimator,feature_count):
        """
        """
        # Get default pipeline step parameters
        _,classifier_grid_parameters,regression_grid_parameters = self.get_default_pipeline_step_parameters(feature_count)

        # Determine estimator type
        if estimator in classifier_grid_parameters:
            estimator_type = 'classifier'
        elif estimator in regression_grid_parameters:
            estimator_type = 'regressor'
        else:
            raise Exception('Estimator %s appears to be unsupported'%(estimator))

        # Return type
        return estimator_type

    def construct_pipeline(self,estimator,
                           feature_selection_type=None,
                           scale_type=None,
                           transform_type=None):
        """
        Returns a sklearn.pipeline.Pipeline object and scoring metric given an estimator argument and optional
        feature selection (feature_selection_type), scaling type (scale_type), and transformation to apply
        (transform_type).
        """
        # Set supported transformers, classifiers, and regressors
        transformers = ['pca','t-sne']

        classifiers = ['knn','logistic_regression','svm','multilayer_perceptron','random_forest','adaboost']

        regressors = ['polynomial_regression']

        estimator_options = classifiers + regressors

        # Raise error if the estimator isn't supported
        assert estimator in estimator_options, 'Estimator %s not available.'%(estimator)

        # Set scoring
        if estimator in classifiers:
            scoring = 'accuracy'
        elif estimator in regressors:
            scoring = 'neg_mean_squared_error'

        # Initialize pipeline steps
        pipeline_steps = []

        # Add feature selection step
        if feature_selection_type:
            if feature_selection_type == 'select_k_best':
                pipeline_steps.append(('feature_selection', SelectKBest(f_classif)))

        # Add scaling step
        if scale_type:
            if scale_type == 'standard':
                pipeline_steps.append(('scaler', StandardScaler()))
            elif scale_type == 'normal':
                pipeline_steps.append(('scaler', Normalizer()))
            elif scale_type == 'min_max':
                pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scale_type == 'binary':
                pipeline_steps.append(('scaler', Binarizer()))

        # Add transforming step
        if transform_type:
            if transform_type == 'pca':
                pipeline_steps.append(('transform', PCA()))
            elif transform_type == 't-sne':
                pipeline_steps.append(('transform', pipeline_TSNE(n_components=2, init='pca')))

        # Add estimator
        if estimator in estimator_options:
            if estimator == 'knn':
                pipeline_steps.append(('estimator', KNeighborsClassifier()))
            elif estimator == 'logistic_regression':
                pipeline_steps.append(('estimator', LogisticRegression()))
            elif estimator == 'svm':
                pipeline_steps.append(('estimator', SVC()))
            elif estimator == 'polynomial_regression':
                pipeline_steps.append(('pre_estimator', PolynomialFeatures()))
                pipeline_steps.append(('estimator', LinearRegression()))
            elif estimator == 'multilayer_perceptron':
                pipeline_steps.append(('estimator', MLPClassifier(solver='lbfgs',alpha=1e-5)))
            elif estimator == 'random_forest':
                pipeline_steps.append(('estimator', RandomForestClassifier()))
            elif estimator == 'adaboost':
                pipeline_steps.append(('estimator', AdaBoostClassifier())) #AdaBoostClassifier(n_estimators=100)
        else:
            error = 'Estimator %s is not recognized. Currently supported estimators are:\n'%(estimator)

            for option in estimator_options:
                error += '\n%s'%(option)

            raise Exception(error)

        # Form/return pipeline and scoring
        return Pipeline(pipeline_steps),scoring

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
                'hidden_layer_sizes': [[x] for x in range(min(3,feature_count),max(3,feature_count)+1)]
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


    def get_parameter_grid(self,estimator,feature_count,
                           feature_selection_type=None,
                           scale_type=None,
                           param_dist=None,
                           use_default_param_dist=True):
        """
        Returns a dictionary of the pipeline step parameters to grid over given the estimator
        (estimator) and number of features (feature_count) arguments and keyword arguments
        representing the type of feature selection (feature_selection_type), type of scaling
        (scale_type),whether default parameters will be used (use_default_param_dist), and a
        user-provided dictionary with pipeline parameters to perform grid search over (param_dist).

        If use_default_param_dist is True and param_dist is provided the default pipeline
        parameters to grid over will be generated yet those in param_dist will override them.

        If use_default_param_dist is False param_dist will be unmodified and will be returned
        as is.
        """
        # Set param_dist to empty dictionary if not given
        if not param_dist:
            param_dist = {}
        else:
            param_dist = dict(param_dist)

        # Add default scaling step to grid parameters
        if use_default_param_dist:
            # Add scaling step
            if scale_type:
                if scale_type == 'min_max':
                    if 'scaler__feature_range' not in param_dist:
                        param_dist['scaler__feature_range'] = [(0,1)]
                elif scale_type == 'binary':
                    if 'scaler__threshold' not in param_dist:
                        param_dist['scaler__threshold'] = [0.5]

            # Add default feature selection parameters
            if feature_selection_type:
                if feature_selection_type == 'select_k_best' and 'feature_selection__k' not in param_dist:
                    """
                    1 feature, 1 interaction degree, 1 combined feature
                    n features, 1 interaction degree, n combined features
                    2 features, 2 interaction degree, 2 + 1 = 3 combined features
                    3 features, 2 interaction degree, 3 + 3 + (4-1) = 9
                    4 features, 2 interactions degree, 4 + 4 + ()
                    """
                    param_dist['feature_selection__k'] = range(1,feature_count+1)

            # Add default estimator parameters
            if estimator == 'knn':
                # Add default number of neighbors for k-nearest neighbors
                if 'estimator__n_neighbors' not in param_dist:
                    param_dist['estimator__n_neighbors'] = range(1,31)

                # Add default point metric options for for k-nearest neighbors
                if 'estimator__weights' not in param_dist:
                    param_dist['estimator__weights'] = ['uniform','distance']
            elif estimator == 'logistic_regression':
                if 'estimator__C' not in param_dist:
                    param_dist['estimator__C'] = np.logspace(-10,10,5)
            elif estimator == 'polynomial_regression':
                if 'pre_estimator__degree' not in param_dist:
                    param_dist['pre_estimator__degree'] = range(1,5)
            elif estimator == 'multilayer_perceptron':
                if 'estimator__hidden_layer_sizes' not in param_dist:
                    param_dist['estimator__hidden_layer_sizes'] = [[x] for x in range(min(3,feature_count),max(3,feature_count)+1)]
            elif estimator == 'random_forest':
                if 'estimator__n_estimators' not in param_dist:
                    param_dist['estimator__n_estimators'] = range(90,100)

        # Return parameters
        return param_dist

    def test_pipeline(self,X_test,y_test,grid_search,estimator_type):
        # Initialize pipeline metrics
        pipeline_metrics = {}

        # Make prediction based on model
        pipeline_metrics['y_pred'] = grid_search.best_estimator_.predict(X_test)

        # Print scores
        if estimator_type == 'classifier':
            # Calculate confusion matrix
            confusion_matrix = sklearn_metrics.confusion_matrix(y_test, pipeline_metrics['y_pred'])

            # Save estimator type
            score_type = 'classification'

            # Calculate training and test scores
            train_score = grid_search.best_score_
            test_score = confusion_matrix.trace()/float(confusion_matrix.sum())

            # Calculate and save normalized confusion matrix
            normalized_confusion_matrix = confusion_matrix/float(confusion_matrix.sum())

            # Save classification report
            classification_report = sklearn_metrics.classification_report(y_test,pipeline_metrics['y_pred'])

            # Save classification metrics
            pipeline_metrics['confusion_matrix'] = confusion_matrix
            pipeline_metrics['normalized_confusion_matrix'] = normalized_confusion_matrix
            pipeline_metrics['classification_report'] = classification_report

        elif estimator_type == 'regressor':
            # Calculate training and test scores
            train_score = -grid_search.best_score_
            test_score = np.sqrt(np.square(y_test-pipeline_metrics['y_pred']).sum(axis=0))

        # Save scores
        pipeline_metrics['test_score'] = test_score
        pipeline_metrics['train_score'] = train_score

        # Return pipeline metrics
        return pipeline_metrics

    def optimize_pipeline(self,X_train,y_train,pipeline,scoring,param_dist,
                          cv=10,
                          n_jobs=-1,
                          num_parameter_combos=None):
        """
        Optimizes provided pipeline using provided data and the grid search cross validation settings
        """
        # Check input types
        assert type(X_train) is np.ndarray,'Input feature values, X, must be of type ndarray'
        assert type(y_train) is np.ndarray,'Input feature values, y, must be of type ndarray'

        # Initialize full or randomized grid search
        if num_parameter_combos:
            grid_search = RandomizedSearchCV(pipeline,
                                             param_dist,
                                             cv=cv,scoring=scoring,
                                             n_iter=num_parameter_combos, # n_iter=10 means 10 random parameter combinations tried
                                             n_jobs=n_jobs)
        else:
            grid_search = GridSearchCV(pipeline,
                                       param_dist,
                                       cv=cv, scoring=scoring, n_jobs=n_jobs)

        # Perform grid search using above parameters
        grid_search.fit(X_train,y_train)

        # Return grid_search object
        return grid_search

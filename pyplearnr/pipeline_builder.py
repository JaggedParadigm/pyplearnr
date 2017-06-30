# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

# Basic tools
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

from sklearn.pipeline import Pipeline

class PipelineBuilder(object):
    """
    Builds a collection of scikit-learn pipelines based on a combinatorial
    schematic.
    """
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

        for step in pipeline_bundle_schematic:
            step_name = list(step.keys())[0]

            step_options = step[step_name]

            step_iterations = []
            for step_option, step_parameters in step_options.items():
                if step_option != 'none':
                    # Get the parameter names for the current step option
                    parameter_names = [parameter_name for parameter_name \
                        in step_parameters.keys()]

                    # Obtain scikit-learn object for the step option
                    if 'sklo' in parameter_names:
                        # Use user-provided object if they mark one of the
                        # step-option parameter names as 'sklo' (scikit-learn
                        # object)
                        sklearn_object = step_parameters['sklo']
                    else:
                        # Use default object if supported
                        if step_option != 'none':
                            sklearn_object = sklearn_packages[step_name][step_option]

                    # Form all parameter combinations for the current step option
                    parameter_combos = [step_parameters[step_parameter_name] \
                                        for step_parameter_name in parameter_names \
                                        if step_parameter_name != 'sklo']

                    # Remove 'sklo'
                    parameter_names = [parameter_name for parameter_name \
                                       in parameter_names \
                                       if parameter_name != 'sklo']

                    # Form all parameter combinations for current step option
                    # and form and append step tuple
                    for parameter_combo in list(itertools.product(*parameter_combos)):

                        parameter_kwargs = {pair[0]: pair[1] \
                                            for pair in zip(parameter_names,
                                                            parameter_combo)}

                        step_addendum = (step_name,
                                         sklearn_object(**parameter_kwargs))

                        step_iterations.append(step_addendum)
                else:
                    # Append nothing if the step is to be ignored
                    step_iterations.append(None)

            pipeline_options.append(step_iterations)

        # Form all step/parameter permutations and convert to scikit-learn
        # pipelines
        pipelines = []
        for pipeline_skeleton in itertools.product(*pipeline_options):
            pipelines.append(Pipeline([step for step in pipeline_skeleton \
                                       if step]))

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
            'pre_estimator': {
                'polynomial_features': PolynomialFeatures
            },
            'estimator': {
                'knn': KNeighborsClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'linear_regression': LinearRegression,
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

# Make compatible with Python 3
from __future__ import print_function

# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2

# Basic tools
import numpy as np
import pandas as pd

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

# Feature selection tools
from sklearn.feature_selection import SelectKBest, f_classif

# Unsupervised learning tools
from sklearn.decomposition import PCA
from sklearn.lda import LDA
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
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

# Classification metrics
import sklearn.metrics as sklearn_metrics

# Define custom TSNE class so that it will work with pipeline
class pipeline_TSNE(TSNE):
    def transform(self,X):
        return self.fit_transform(X)

class OptimizedPipeline:
    """
    Pipeline with optimal parameters found through nested k-folds cross validation
    """
    def __init__(self,estimator,
                 feature_selection_type=None,
                 scale_type=None,
                 feature_interactions=False,
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
        
        self.feature_interactions = feature_interactions # Whether feature interactions are used
        
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
                                                             transform_type=self.transform_type,
                                                             feature_interactions=self.feature_interactions)
        
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
                           transform_type=None,
                           feature_interactions=False):
        """
        Returns a sklearn.pipeline.Pipeline object and scoring metric given an estimator argument and optional
        feature selection (feature_selection_type), scaling type (scale_type), transformation to apply
        (transform_type), and whether feature_interactions are desired.
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
                
        # Add feature interactions
        if feature_interactions:
            pipeline_steps.append(('feature_interactions',PolynomialFeatures()))
            
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
            'feature_interactions': {
                'degree': range(1,3)
            },
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
                           feature_interactions=False,
                           param_dist=None,
                           use_default_param_dist=True):
        """
        Returns a dictionary of the pipeline step parameters to grid over given the estimator
        (estimator) and number of features (feature_count) arguments and keyword arguments
        representing the type of feature selection (feature_selection_type), type of scaling
        (scale_type), whether feature interactions are to be used (feature_interactions),
        whether default parameters will be used (use_default_param_dist), and a user-provided
        dictionary with pipeline parameters to perform grid search over (param_dist).
        
        If use_default_param_dist is True and param_dist is provided the default pipeline
        parameters to grid over will be generated yet those in param_dist will override them.
        
        If use_default_param_dist is False param_dist will be unmodified and will be returned
        as is.
        """
        # Set param_dist to empty dictionary if not given
        if not param_dist:
            param_dist = {}
            
        # Add default scaling step to grid parameters
        if use_default_param_dist:
            # Add default feature interaction parameters
            if feature_interactions:
                if 'feature_interactions__degree' not in param_dist:
                    param_dist['feature_interactions__degree'] = range(1,3)
                        
            # Add default feature selection parameters
            if feature_selection_type:
                if feature_selection_type == 'select_k_best' and 'feature_selection__k' not in param_dist:
                    if not feature_interactions:
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
                X = X.values
            else:
                raise Exception('Data input, X, must be of type pandas.core.frame.DataFrame, \
                                pandas.core.series.Series, or numpy.ndarray')
        
        if type(y) is not np.ndarray:        
            if type(y) is pd.core.frame.DataFrame or type(y) is pd.core.series.Series:
                y = y.values
            else:
                raise Exception('Data output, y, must be of type pandas.core.frame.DataFrame, \
                                pandas.core.series.Series, or numpy.ndarray')
        
        # Save data
        self.X = X.copy()
        self.y = y.copy()
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X,y,test_size=test_size,random_state=random_state)
        
        # Save cross-validation settings
        self.cv = cv
        self.num_parameter_combos = num_parameter_combos
        
        # Save number of features
        self.feature_count = self.X.shape[1]
        
        # Determine estimator type
        self.estimator_type = self.get_estimator_type(self.estimator,self.feature_count)        
        
        # Derive pipeline parameters to grid over
        self.param_dist = self.get_parameter_grid(self.estimator,self.feature_count,
                                                  feature_selection_type=self.feature_selection_type,
                                                  scale_type=self.scale_type,
                                                  feature_interactions=self.feature_interactions,
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
                                       cv=cv, scoring=scoring,n_jobs=n_jobs)
        
        # Perform grid search using above parameters
        grid_search.fit(X_train,y_train)
        
        # Return grid_search object
        return grid_search
        
#def train_model(X,y,
#               scale_type=None,
#               feature_interactions=False,
#               transform_type=None,
#               feature_selection_type=None,
#               estimator='knn',
#               param_dist=None,
#               use_default_param_dist=False,
#               n_jobs=-1,
#               num_parameter_combos=[],
#               cv=10,
#               random_state=[],
#               suppress_output=False):
#    """
#    """
#    # Set param_dist to empty dictionary if not given
#    if not param_dist:
#        param_dist = {}
#    
#    # Convert X and y to ndarray if either Pandas series or dataframe
#    if type(X) is not np.ndarray:
#        if type(X) is pd.core.frame.DataFrame or type(X) is pd.core.series.Series:
#            X = X.values
#        else:
#            raise Exception('Data input, X, must be of type pandas.core.frame.DataFrame, \
#                            pandas.core.series.Series, or numpy.ndarray')
#    
#    if type(y) is not np.ndarray:        
#        if type(y) is pd.core.frame.DataFrame or type(y) is pd.core.series.Series:
#            y = y.values
#        else:
#            raise Exception('Data output, y, must be of type pandas.core.frame.DataFrame, \
#                            pandas.core.series.Series, or numpy.ndarray')
#        
#    # Get number of features
#    num_features = X.shape[1]
#        
#    # Set classifiers
#    classifiers = ['knn','logistic_regression','svm','multilayer_perceptron','random_forest','adaboost']
#    
#    # Set regressors
#    regressors = ['polynomial_regression']
#    
#    # Set estimator options
#    estimator_options = classifiers + regressors
#    
#    # Set transformers
#    transformers = ['pca','t-sne']
#
#    # Initialize pipeline steps
#    pipeline_steps = []
#
#    # Add feature selection step
#    if feature_selection_type:
#        if feature_selection_type == 'select_k_best':
#            pipeline_steps.append(('feature_selection', SelectKBest(f_classif)))
#    
#    # Add scaling step
#    if scale_type:
#        if scale_type == 'standard':
#            pipeline_steps.append(('scaler', StandardScaler()))
#            
#    # Add feature interactions
#    if feature_interactions:
#        pipeline_steps.append(('feature_interactions',PolynomialFeatures()))
#        
#    # Add transforming step
#    if transform_type:
#        if transform_type == 'pca':
#            pipeline_steps.append(('transform', PCA()))
#        elif transform_type == 't-sne':            
#            pipeline_steps.append(('transform', pipeline_TSNE(n_components=2, init='pca')))
#    
#    # Add estimator
#    if estimator in estimator_options:
#        if estimator == 'knn':
#            pipeline_steps.append(('estimator', KNeighborsClassifier()))
#        elif estimator == 'logistic_regression':
#            pipeline_steps.append(('estimator', LogisticRegression()))
#        elif estimator == 'svm':
#            pipeline_steps.append(('estimator', SVC()))
#        elif estimator == 'polynomial_regression':
#            pipeline_steps.append(('pre_estimator', PolynomialFeatures()))
#            pipeline_steps.append(('estimator', LinearRegression()))
#        elif estimator == 'multilayer_perceptron':
#            pipeline_steps.append(('estimator', MLPClassifier(solver='lbfgs',alpha=1e-5)))
#        elif estimator == 'random_forest':
#            pipeline_steps.append(('estimator', RandomForestClassifier()))
#        elif estimator == 'adaboost':
#            pipeline_steps.append(('estimator', AdaBoostClassifier())) #AdaBoostClassifier(n_estimators=100)                        
#    else:
#        error = 'Estimator %s is not recognized. Currently supported estimators are:\n'%(estimator)
#
#        for option in estimator_options:
#            error += '\n%s'%(option)
#
#        raise Exception(error)
#
#    # Add default scaling step to grid parameters
#    if use_default_param_dist:
#        # Add default feature interaction parameters
#        if feature_interactions:
#            if 'feature_interactions__degree' not in param_dist:
#                param_dist['feature_interactions__degree'] = range(1,3)
#                    
#        # Add default feature selection parameters
#        if feature_selection_type:
#            if feature_selection_type == 'select_k_best' and 'feature_selection__k' not in param_dist:
#                if not feature_interactions:
#                    param_dist['feature_selection__k'] = range(1,num_features+1)
#        
#        # Add default estimator parameters
#        if estimator == 'knn':
#            # Add default number of neighbors for k-nearest neighbors
#            if 'estimator__n_neighbors' not in param_dist:
#                param_dist['estimator__n_neighbors'] = range(1,31)
#                
#            # Add default point metric options for for k-nearest neighbors
#            if 'estimator__weights' not in param_dist:
#                param_dist['estimator__weights'] = ['uniform','distance']
#        elif estimator == 'logistic_regression':
#            if 'estimator__C' not in param_dist:
#                param_dist['estimator__C'] = np.logspace(-10,10,5)
#        elif estimator == 'polynomial_regression':
#            if 'pre_estimator__degree' not in param_dist:
#                param_dist['pre_estimator__degree'] = range(1,5)
#        elif estimator == 'multilayer_perceptron':
#            if 'estimator__hidden_layer_sizes' not in param_dist: 
#                param_dist['estimator__hidden_layer_sizes'] = [[x] for x in range(min(3,num_features),max(3,num_features)+1)]
#        elif estimator == 'random_forest':
#            if 'estimator__n_estimators' not in param_dist:
#                param_dist['estimator__n_estimators'] = range(90,100)
#
#    # Form pipeline
#    pipeline = Pipeline(pipeline_steps)
#    
#    # Initialize extra fields
#    pipeline.score_type = [] # "classification" or "regression" score
#    pipeline.train_score = [] # Score from training set
#    pipeline.test_score = [] # Score from test set        
#    pipeline.confusion_matrix = [] # Confusion matrix, if classification
#    pipeline.normalized_confusion_matrix = [] # Confusion matrix divided by its total sum
#    pipeline.classification_report = [] # Classification report
#    pipeline.best_parameters = [] # Best parameters            
#    
#    # Set scoring
#    if estimator in classifiers:
#        scoring = 'accuracy'
#    elif estimator in regressors:
#        scoring = 'neg_mean_squared_error'
#        
#    # Print grid parameters
#    if not suppress_output:
#        print('Grid parameters:')
#        for x in param_dist:
#            print(x,':',param_dist[x])
#    
#    # Initialize full or randomized grid search
#    if num_parameter_combos:
#        grid_search = RandomizedSearchCV(pipeline,
#                                         param_dist,
#                                         cv=cv,scoring=scoring,
#                                         n_iter=num_parameter_combos, # n_iter=10 means 10 random parameter combinations tried
#                                         n_jobs=n_jobs)
#    else:
#        grid_search = GridSearchCV(pipeline, 
#                                   param_dist,
#                                   cv=cv, scoring=scoring,n_jobs=n_jobs)
#    
#    # Split data into train and test sets
#    X_train, X_test, y_train, y_test = \
#        train_test_split(X,y,test_size=0.2,random_state=random_state)
#    
#    # Perform grid search using above parameters
#    grid_search.fit(X_train,y_train)
#    
#    # Make prediction based on model
#    y_pred = grid_search.best_estimator_.predict(X_test)
#
#    # Print scores
#    if estimator in classifiers:
#        # Calculate confusion matrix
#        pipeline.confusion_matrix = confusion_matrix(y_test, y_pred)
#        
#        # Save estimator type
#        pipeline.score_type = 'classification'
#        
#        # Get and save training score
#        pipeline.train_score = grid_search.best_score_
#        
#        # Get and save test score
#        pipeline.test_score = pipeline.confusion_matrix.trace()/float(pipeline.confusion_matrix.sum())
#        
#        # Calculate and save normalized confusion matrix
#        pipeline.normalized_confusion_matrix = pipeline.confusion_matrix/float(pipeline.confusion_matrix.sum())
#        
#        # Save classification report
#        pipeline.classification_report = classification_report(y_test, y_pred)        
#        
#        # Print output if desired
#        if not suppress_output:
#            # Print training and test scores
#            print('\nTraining set classification accuracy: ', pipeline.train_score)
#            print('\nTest set classification accuracy: ', pipeline.test_score)
#            
#            # Print out confusion matrix and normalized confusion matrix containing probabilities
#            print('Confusion matrix: \n\n',pipeline.confusion_matrix)
#            print('\nNormalized confusion matrix: \n\n', pipeline.normalized_confusion_matrix)
#    
#            # Print out classification report
#            print('\nClassification report: \n\n',pipeline.classification_report)
#    elif estimator in regressors:
#        # Save estimator type
#        pipeline.score_type = 'regression'
#        
#        # Calculate and save training and test scores
#        pipeline.train_score = -grid_search.best_score_
#        pipeline.test_score = np.sqrt(np.square(y_test-y_pred).sum(axis=0))
#        
#        # Print output if desired
#        if not suppress_output:
#            print('\nTraining L2 norm score: ',pipeline.train_score)
#            print('\nTest L2 norm score: ',pipeline.test_score)
#    
#    # Save best parameters
#    pipeline.best_parameters = grid_search.best_params_
#    
#    # Print best parameters
#    if not suppress_output:
#        print('\nBest parameters:\n')
#        print(pipeline.best_parameters)
#
#    # Fit pipeline with best parameters obtained from grid search using all data
#    pipeline.set_params(**grid_search.best_params_).fit(X, y)
#
#    # Print pipeline
#    if not suppress_output:
#        print('\n',pipeline)
#    
#    # Return pipeline
#    return pipeline # Use pipeline.predict() for productionalization
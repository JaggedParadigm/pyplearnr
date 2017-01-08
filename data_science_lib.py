# Make compatible with Python 3
from __future__ import print_function

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define custom TSNE class so that it will work with pipeline
class pipeline_TSNE(TSNE):
    def transform(self,X):
        return self.fit_transform(X)
    
def train_model(X,y,
               scale_type=None,
               feature_interactions=False,
               transform_type=None,
               feature_selection_type=None,
               estimator='knn',
               param_dist=None,
               use_default_param_dist=False,
               n_jobs=-1,
               num_parameter_combos=[],
               cv=10,
               random_state=[],
               suppress_output=False):
    """
    """
    # Set param_dist to empty dictionary if not given
    if not param_dist:
        param_dist = {}
    
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
        
    # Get number of features
    num_features = X.shape[1]
        
    # Set classifiers
    classifiers = ['knn','logistic_regression','svm','multilayer_perceptron','random_forest','adaboost']
    
    # Set regressors
    regressors = ['polynomial_regression']
    
    # Set estimator options
    estimator_options = classifiers + regressors
    
    # Set transformers
    transformers = ['pca','t-sne']

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
            #pipeline_steps.append(('transform', TSNE(n_components=2, init='pca')))
                            
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
                    param_dist['feature_selection__k'] = range(1,num_features+1)
        
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
                param_dist['estimator__hidden_layer_sizes'] = [[x] for x in range(min(3,num_features),max(3,num_features)+1)]
        elif estimator == 'random_forest':
            if 'estimator__n_estimators' not in param_dist:
                param_dist['estimator__n_estimators'] = range(90,100)

    # Form pipeline
    pipeline = Pipeline(pipeline_steps)
    
    # Initialize extra fields
    pipeline.score_type = [] # "classification" or "regression" score
    pipeline.train_score = [] # Score from training set
    pipeline.test_score = [] # Score from test set        
    pipeline.confusion_matrix = [] # Confusion matrix, if classification
    pipeline.normalized_confusion_matrix = [] # Confusion matrix divided by its total sum
    pipeline.classification_report = [] # Classification report
    pipeline.best_parameters = [] # Best parameters            
    
    # Set scoring
    if estimator in classifiers:
        scoring = 'accuracy'
    elif estimator in regressors:
        scoring = 'neg_mean_squared_error'
        
    # Print grid parameters
    if not suppress_output:
        print('Grid parameters:')
        for x in param_dist:
            print(x,':',param_dist[x])
    
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
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X,y,test_size=0.2,random_state=random_state)
    
    # Perform grid search using above parameters
    grid_search.fit(X_train,y_train)
    
    # Make prediction based on model
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Print scores
    if estimator in classifiers:
        # Calculate confusion matrix
        pipeline.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Save estimator type
        pipeline.score_type = 'classification'
        
        # Get and save training score
        pipeline.train_score = grid_search.best_score_
        
        # Get and save test score
        pipeline.test_score = pipeline.confusion_matrix.trace()/float(pipeline.confusion_matrix.sum())
        
        # Calculate and save normalized confusion matrix
        pipeline.normalized_confusion_matrix = pipeline.confusion_matrix/float(pipeline.confusion_matrix.sum())
        
        # Save classification report
        pipeline.classification_report = classification_report(y_test, y_pred)        
        
        # Print output if desired
        if not suppress_output:
            # Print training and test scores
            print('\nTraining set classification accuracy: ', pipeline.train_score)
            print('\nTest set classification accuracy: ', pipeline.test_score)
            
            # Print out confusion matrix and normalized confusion matrix containing probabilities
            print('Confusion matrix: \n\n',pipeline.confusion_matrix)
            print('\nNormalized confusion matrix: \n\n', pipeline.normalized_confusion_matrix)
    
            # Print out classification report
            print('\nClassification report: \n\n',pipeline.classification_report)
    elif estimator in regressors:
        # Save estimator type
        pipeline.score_type = 'regression'
        
        # Calculate and save training and test scores
        pipeline.train_score = -grid_search.best_score_
        pipeline.test_score = np.sqrt(np.square(y_test-y_pred).sum(axis=0))
        
        # Print output if desired
        if not suppress_output:
            print('\nTraining L2 norm score: ',pipeline.train_score)
            print('\nTest L2 norm score: ',pipeline.test_score)
    
    # Save best parameters
    pipeline.best_parameters = grid_search.best_params_
    
    # Print best parameters
    if not suppress_output:
        print('\nBest parameters:\n')
        print(pipeline.best_parameters)

    # Fit pipeline with best parameters obtained from grid search using all data
    pipeline.set_params(**grid_search.best_params_).fit(X, y)

    # Print pipeline
    if not suppress_output:
        print('\n',pipeline)
    
    # Return pipeline
    return pipeline # Use pipeline.predict() for productionalization
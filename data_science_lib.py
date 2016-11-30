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

def train_model(X,y,
               scale_type=None,
               transform_type=None,
               feature_selection_type=None,
               estimator='knn',
               param_dist={},
               use_default_param_dist=False,
               n_jobs=-1,
               num_parameter_combos=[],
               cv=10):
    """
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
        
    # Get number of features
    num_features = X.shape[1]
        
    # Set classifiers
    classifiers = ['knn','logistic_regression','svm','multilayer_perceptron','random_forest']
    
    # Set regressors
    regressors = ['polynomial_regression']
    
    # Set estimator options
    estimator_options = classifiers + regressors
    
    # Set transformers
    transformers = ['pca','t-sne']

    # Initialize pipeline steps
    pipeline_steps = []
    
    # Add scaling step
    if scale_type:
        if scale_type == 'standard':
            pipeline_steps.append(('scaler', StandardScaler()))
            
    # Add transforming step
    if transform_type:
        if transform_type == 'pca':
            pipeline_steps.append(('transform', PCA()))
        elif transform_type == 't-sne':
            pipeline_steps.append(('transform', TSNE(n_components=2, init='pca')))
                
    # Add feature selection step
    if feature_selection_type:
        if feature_selection_type == 'select_k_best':
            pipeline_steps.append(('feature_selection', SelectKBest(f_classif)))
            
    # Add estimator
    if estimator in estimator_options:
        if estimator == 'knn':
            pipeline_steps.append(('estimator', KNeighborsClassifier()))
        elif estimator == 'logistic_regression':
            pipeline_steps.append(('estimator', LogisticRegression()))
        elif estimator == 'svm':
            pipeline_steps.append(('estimator', SVC()))
        elif estimator == 'polynomial_regression':
            pipeline_steps.append(('pre_estimator',PolynomialFeatures()))
            pipeline_steps.append(('estimator', LinearRegression()))
        elif estimator == 'multilayer_perceptron':
            pipeline_steps.append(('estimator', MLPClassifier(solver='lbfgs',alpha=1e-5)))
        elif estimator == 'random_forest':
            pipeline_steps.append(('estimator', RandomForestClassifier()))
    else:
        error = 'Estimator %s is not recognized. Currently supported estimators are:\n'%(estimator)

        for option in estimator_options:
            error += '\n%s'%(option)

        raise Exception(error)

    # Add default scaling step to grid parameters
    if use_default_param_dist:
        # Add default feature selection parameters
        if feature_selection_type:
            if feature_selection_type == 'select_k_best' and 'feature_selection__k' not in param_dist:
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
    
    # Set scoring
    if estimator in classifiers:
        scoring = 'accuracy'
    elif estimator in regressors:
        scoring = 'neg_mean_squared_error'
        
    # Print grid parameters
    print 'Grid parameters:'
    for x in param_dist:
        print x,':',param_dist[x]
    
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
        train_test_split(X, y, test_size=0.2)

    # Perform grid search using above parameters
    grid_search.fit(X_train,y_train)

    # Make prediction based on model
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Print scores
    if estimator in classifiers:
        # Calculate confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)

        # Print training and test scores
        print '\nTraining set classification accuracy: ',grid_search.best_score_
        print '\nTest set classification accuracy: ',conf_mat.trace()/float(conf_mat.sum())
        
        # Print out confusion matrix and normalized confusion matrix containing probabilities
        print 'Confusion matrix: \n\n',conf_mat
        print '\nNormalized confusion matrix: \n\n',conf_mat/float(conf_mat.sum())

        # Print out classification report
        print '\nClassification report: \n\n',classification_report(y_test, y_pred)        
    elif estimator in regressors:
        print '\nTraining L2 norm score: ',-grid_search.best_score_
        print '\nTest L2 norm score: ',np.sqrt(np.square(y_test-y_pred).sum(axis=0))
    
    # Print best parameters
    print '\nBest parameters:\n'
    print grid_search.best_params_

    # Fit pipeline with best parameters obtained from grid search using all data
    pipeline.set_params(**grid_search.best_params_).fit(X, y)

    # Print pipeline
    print '\n',pipeline
    
    # Return pipeline
    return pipeline # Use pipeline.predict() for productionalization
        

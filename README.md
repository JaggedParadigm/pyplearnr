# How
## Installation
For now, simply clone the respository, link to the location in your code, and import it. 

## Use
See the [demo](https://nbviewer.jupyter.org/github/JaggedParadigm/pyplearnr/blob/master/pyplearnr_demo.ipynb) for use of pyplearnr.

# What
Pyplearnr is a tool designed to easily and more elegantly build, validate, and test scikit-learn pipelines.

One core aspect of pyplearnr is the combinatorial pipeline schematic, a flexible diagram of every step (e.g. estimator), step option (e.g. knn, logistic regression, etc.), and parameter option (e.g. n_neighbors for knn and C for logistic regression) combination. Any scikit-learn class instance you would use in a normal pipeline can be inserted or one can be chosen from a list of supported ones. 

Here's an example with optional scaling, PCA, selection of the number of principal components to use, and the use of k-nearest neighbors with different values for the number of neighbors:
```python
pipeline_schematic = [
    {'scaler': {
            'min_max': {},
            'standard': {}
        }
    },
    {'transform': {
            'pca': {
                'n_components': [feature_count]
            }
        }         
    },
    {'feature_selection': {
            'select_k_best': {
                'k': range(1, feature_count+1)
            }
        }
    },
    {'estimator': {
            'knn': {
                'n_neighbors': range(1,31)
                }
        }
    }
]
```

The core validation method is nested k-fold cross-validation (stratified if for classification). Pyplearnr divides the data into k validation outer-folds and their corresponding training sets into k test inner-folds, picks the best pipeline as that having the highest score (median by default) for the inner-folds for each outer-fold, chooses the winning pipeline as that with the most wins, and uses the validation outer-folds to give an estimate of the ultimate winner's out-of-sample scores. This final pipeline can then be used to make predictions.

# Why
I wanted to a way to do what GridSearchCV does for specific estimators with any estimator in a repeatable way.
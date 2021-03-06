# What
Pyplearnr is a tool designed to perform model selection, hyperparameter tuning, and model validation via nested k-fold cross-validation in a reproducible way.

# Why
I found GridSearchCV to be lacking. I wanted a tool that used a similar procedure to perform simultaneous hyperparameter tuning AND model selection with a clear input that summarizes exactly what scikit-learn pipeline steps and parameter combinations will used and whose results allow perfect reproducibility. So, I made my own.

# How
### Use
See the [demo](https://nbviewer.jupyter.org/github/JaggedParadigm/pyplearnr/blob/master/pyplearnr_demo.ipynb) for more detailed use of pyplearnr with actual data.

Here are the basic steps:
#### 1) Place feature data into non-null feature matrix and target vector
#### 2) Initialize the nested k-fold cross-validation object
```python
kfcv = ppl.NestedKFoldCrossValidation(outer_loop_fold_count=5, 
                                      inner_loop_fold_count=5)
```
#### 3) Specify the combinatorial pipeline schematic detailing all possible model/parameter combinations 

Ex: Here's an example of model/parameter combinations of optional scaling of two types, a principal component analysis directly using scikit-learn's sklearn.decomposition.PCA transformer, selection of data transformed by k principal components (between 1 and 30), and the use of either a k-nearest neighbors classifier (k between 1 and 30) or random forest classifier with a maximum depth between 2 and 5 (and a specified random state for reproducibility).

```python
pipeline_schematic = [
    {'scaler': {
            'none': {},
            'min_max': {},
            'standard': {}
        }
    },
    {'transform': {
            'pca': {
                'sklo': sklearn.decomposition.PCA,
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
            },
            'random_forest': {
                'sklo': RandomForestClassifier,
                'max_depth': range(2,6),
                'random_state': [57]
			}
        }
    }
]
```

#### 4) Run pyplearnr
```python
# Perform nested k-fold cross-validation
kfcv.fit(X, y, pipeline_schematic=pipeline_schematic, 
         scoring_metric='auc', score_type='median')
```
### Methodology
The core model selection and validation method is nested k-fold cross-validation (stratified if for classification). Inner-fold contests are used for model selection and outer-folds are used to cross-validate the final winning model. 

Here's the basic algorithm used by pyplearnr:

- 1) Pyplearnr shuffles and divides the data into k validation outer-folds. 
- 2) For each outer-fold:
	- a) The remaining folds are combined to form the corresponding training set
	- b)  This training set is divided into k (or possibly a different number) of inner-test-folds.
	- c) For each inner-test-fold:
	  - i) The remaining inner-test-folds are combined and used to train all pipelines/models, which are scored on the corresponding inner-test-fold
  - d) The winning model/pipeline of each inner-test-fold contest is chosen as that with the best median score over all inner-test-folds
	  - iii) The user is alerted If there is a tie and expected to decide the winning pipeline (usually the simplest for better generalizability)
- 4) The final winning model/pipeline is chosen as that with the most number of wins from all inner-test-fold contests corresponding to each outer-fold 
	- e) Again, the user is expected to decide the winner If there is a tie
- 5) This final winning model/pipeline is trained on all of the training data for each outer-fold, tested on the corresponding validation set, and summary statistics are presented to the user representing expected out-of-sample performance.


### Installation
##### Dependencies

pyplearnr requires:

Python (>= 2.7 or >= 3.3)
scikit-learn (>= 0.18.2)
numpy (>= 1.13.0)
scipy (>= 0.19.1)
pandas (>= 0.20.2)
matplotlib (>= 2.0.2)

For use in Jupyter notebooks and the conda installation, I recommend having nb_conda (>= 2.2.0).

### User installation
Install by using pip:

```
pip install pyplearnr
```

For conda, you can issue the same command above within a conda environment or you can include this in your environment.yml file:

```
- pip:
    - pyplearnr
```

and then either generate a new environment from the terminal using:

```
conda env create
```

or update an existing one (environment_name) using:

```
conda env update -n=environment_name -f=./environment.yml
```

Another option is to simply clone the respository, link to the location in your code, and import it. 



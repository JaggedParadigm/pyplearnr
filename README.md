# How
## Installation
For now, simply clone the respository, link to the location in your code, and import it. 

## Use
See the [demo](pyplearnr_demo.ipynb) for use of pyplearnr.

# What
Pyplearnr is a way to build, validate, and test multiple scikit learn pipelines, with varying steps, in a less verbose manner.

Quick keyword arguments give access to optional feature selection (e.g. SelectKBest), scaling (e.g. standard scaling), use of feature interactions, and data transformations (e.g. PCA, t-SNE) before being fed to a classifier/regressor.

After building the pipeline, data can be used to perform a nested (stratified if classification) k-folds cross-validation and output an object containing data from the process, including the best model.

Various default pipeline step parameters for the grid-search are available for quick iteration over different pipelines, with the option to ignore/override them in a flexible way.

This is an on-going project that I intend to update with more models and pre-processing options and also with corresponding defaults.

# Why
I wanted to a way to quickly find the best cross-validated model for a given dataset using convenient grid-search parameter defaults in the most succinct manner possible.
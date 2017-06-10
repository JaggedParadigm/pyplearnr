import numpy as np

from .pipeline_evaluator import PipelineEvaluator

class TrainedPipeline(object):
    """
    Class mostly for storing metadata on pipelines (at least for the moment)
    """
    def __init__(self, pipeline_id=None, pipeline=None, scoring_metric=None,
                 score_type='median'):
        ############### Initialize fields ###############
        self.id = pipeline_id

        self.pipeline = pipeline

        self.estimator_type = None

        # Whether a median and IQR or mean and std metric are used for
        # statistics
        self.score_type = score_type

        self.test_scores = []
        self.train_scores = []

        # Metric ('rmse', 'accuracy', 'auc') used to score pipelines
        self.scoring_metric = scoring_metric

        # Data related
        self.X_train = None
        self.y_train = None
        self.y_test_pred = None

        self.X_test = None
        self.y_test = None
        self.y_train_pred = None

        ############### Infer fields ###############
        if self.scoring_metric == 'rmse':
            self.estimator_type = 'regressor'
        else:
            self.estimator_type = 'classification'

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Simple wrapper for the fit method of the pipeline
        """
        ############### Save inputs ###############
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        ############### Fit on training data ###############
        # Fit input data to individual pipeline
        self.pipeline.fit(X_train, y_train)

        ############### Get predicted targets ###############
        self.y_test_pred = self.predict(X_test)

        self.y_train_pred = self.predict(X_train)

        ############### Obtain relevant scores ###############
        evaluator = PipelineEvaluator()

        # Calculate train/test scores
        self.train_scores.append(evaluator.get_score(self.y_train,
                                                     self.y_train_pred,
                                                     self.scoring_metric))

        self.test_scores.append(evaluator.get_score(self.y_test,
                                                    self.y_test_pred,
                                                    self.scoring_metric))

    def predict(self, X):
        """
        Predicts targets given a feature array
        """
        return self.pipeline.predict(X)

class OuterFoldTrainedPipeline(TrainedPipeline):
    """
    Class for pipelines contained in outer loops
    """
    def __init__(self, pipeline_id=None, pipeline=None, scoring_metric=None,
                 score_type='median'):
        # Tell class it is itself for weird Jupyter notebook %autoload
        # incompatibility
        self.__class__ = OuterFoldTrainedPipeline

        super(OuterFoldTrainedPipeline, self).__init__(
            pipeline_id=pipeline_id,
            pipeline=pipeline,
            scoring_metric=scoring_metric,
            score_type=score_type)

        ############### Initialize addtional fields ###############
        self.inner_loop_test_scores = None
        self.inner_loop_train_scores = None

    def set_inner_loop_scores(self, train_scores, test_scores):
        self.inner_loop_train_scores = train_scores
        self.inner_loop_test_scores = test_scores

    def get_inner_loop_score_center(self, score_type=None, fold_type=None):
        """
        Returns the measure of centrality of the inner-fold test or train
        scores

        score_type :    str, {'mean', 'median'}, optional
            Statistical measure to return
            mean :  Use the highest mean of the inner-fold test scores to pick
                    the best pipeline/model
            median :    Use the highest median of the inner-fold test scores to
                        pick the best pipeline/model

        fold_type : str, {'test', 'train'}
            Type of score to obtain for designated pipeline
            test: Score(s) for test fold
            train: Score(s) for train fold

        """
        ############### Check inputs ###############
        if not score_type or type(score_type) is not str \
            or score_type not in ['mean', 'median']:

            raise Exception("The keyword argument dictating the test " \
                            "statistic to return, score_type, must be either" \
                            " 'mean' or 'median'")

        if not fold_type or type(fold_type) is not str \
            or fold_type not in ['test', 'train']:

            raise Exception("The keyword argument indicating the type of " \
                            "fold, fold_type, must be 'test or 'train'")

        # Get scores
        if fold_type == 'test':
            fold_scores = self.inner_loop_test_scores
        elif fold_type == 'train':
            fold_scores = self.inner_loop_train_scores

        # Get measure of centrality
        if score_type == 'mean':
            centrality_measure = np.mean(fold_scores)
        elif score_type == 'median':
            centrality_measure = np.median(fold_scores)

        return centrality_measure

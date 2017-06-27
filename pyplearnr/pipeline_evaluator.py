# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

import numpy as np

# Classification metrics
import sklearn.metrics as sklearn_metrics

from sklearn.metrics import classification_report

class PipelineEvaluator(object):
    """
    Class used to evaluate pipelines
    """
    def get_score(self, y, y_pred, scoring_metric):
        """
        Returns the score given target values, predicted values, and a scoring
        metric.

        Parameters
        ----------
        y : numpy.array
            Actual target array

        y_pred :    numpy.array
            Predicted target array

        scoring_metric :    str, {'auc'}
            Metric used to score estimator

        Returns
        -------
        score : floats
            Score of type scoring_metric determined from the actual and
            predicted target values

        """
        ############### Check inputs ###############
        if not self.metric_supported(scoring_metric):
            raise Exception("The third positional argumet, indicating the " \
                            "estimator scoring metric, %s, is currently "\
                            "unsupported"%(scoring_metric))

        if type(y) is not np.ndarray or type(y_pred) is not np.ndarray:
            raise Exception("The 1st and 2nd positional arguments, " \
                            "representing the respective actual and " \
                            "predicted target arrays, must be of type " \
                            "numpy.array")

        if len(y.shape) != 1 or len(y_pred.shape) != 1 \
            or y.shape[0] != y_pred.shape[0]:

            raise Exception("The 1st and 2nd positional arguments, " \
                            "representing the respective actual and " \
                            "predicted target arrays, must both be of shape " \
                            "(m, )")

        y, y_pred = self.remove_nan(y, y_pred)

        ############### Calculate score ###############
        if scoring_metric == 'auc':
            # Get ROC curve points
            false_positive_rate, true_positive_rate, _ = \
                sklearn_metrics.roc_curve(y, y_pred)

            # Calculate the area under the curve
            score =  sklearn_metrics.auc(false_positive_rate, true_positive_rate)
        elif scoring_metric == 'rmse':
            # Calculate the root mean square error
            score = -np.sqrt(sklearn_metrics.mean_squared_error(y, y_pred))

        elif scoring_metric == 'accuracy':
            score = sklearn_metrics.accuracy_score(y, y_pred)

        return score

    def remove_nan(self, y, y_pred):
        """
        Removes rows from y and y_pred if either have nan values
        """
        kept_inds = [row_ind for row_ind in range(y.shape[0]) if ~np.isnan(y_pred[row_ind])]

        no_nan_y = y[kept_inds]
        no_nan_y_pred = y_pred[kept_inds]

        return no_nan_y, no_nan_y_pred

    def metric_supported(self, metric):
        """
        Tells whether estimator scoring metric (Ex: 'auc') is currently
        supported

        Parameters
        ----------
        metric :    str

        Returns
        -------
        support_flag :  boolean
            True :  if the metric is supported
            False : if the metric is not supported

        """
        supported_metrics = ['auc', 'accuracy', 'rmse']

        if metric:
            if metric in supported_metrics:
                support_flag = True
            else:
                support_flag = False
        else:
            raise Exception("The first positional argument must be an " \
                            "estimator scoring metric (Ex: 'auc')")

        return support_flag

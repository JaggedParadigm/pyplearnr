# Author: Christopher M. Shymansky <CMShymansky@gmail.com>,
# License: ALv2
# Date created: 2016-11-25

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
        """
        if scoring_metric == 'auc':
            # Get ROC curve points
            false_positive_rate, true_positive_rate, _ = sklearn_metrics.roc_curve(y,
                                                                                   y_pred)

            # Calculate the area under the curve
            score =  sklearn_metrics.auc(false_positive_rate, true_positive_rate)

        return score

    def get_classification_report(self, y, y_pred):
        return classification_report(y, y_pred) 

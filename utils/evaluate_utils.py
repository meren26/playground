from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
import numpy as np
from sklearn.model_selection import cross_val_score


def get_classification_scores(classifier, X_test, y_test):
    """
    Returns classification evaluations.

    classifier: cf object
    X_test, y_test
    """
    # accuracy = TP + TN / (TP + TN + FP + FN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)  aka sensitivity, or true positive rate
    # f1 = 2 * precision * recall / (precision + recall)

    preds = classifier.predict(X_test)

    print('accuracy = ', accuracy_score(y_test, preds))
    print('precision = ', precision_score(y_test, preds))
    print('recall = ', recall_score(y_test, preds))
    print('f1 = ', f1_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

def get_regression_scores(regressor, X_test, y_test):
    """
    Returns regression evaluations.

    regressor: rg object
    X_test, y_test
    """

    preds = regressor.predict(X_test)

    print('MAE = ', mean_absolute_error(y_test, preds))
    print('MSE = ', mean_squared_error(y_test, preds))
    print('RMSE = ', np.sqrt(mean_squared_error(y_test, preds)))
    print('R2 = ', r2_score(y_test, preds))

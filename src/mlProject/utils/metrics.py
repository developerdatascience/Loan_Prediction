import pandas as pd
import numpy as np
from src.mlProject import logger
from sklearn.metrics import (f1_score, 
                             accuracy_score, 
                             recall_score, 
                             precision_score, 
                             roc_auc_score,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
from typing import Dict, Any, Union


def calculate_model_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "f1_score_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy_score": float(accuracy_score(y_true, y_pred)),
        "recall_score_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "precision_score": float(precision_score(y_true, y_pred, average="weighted")),
    }

def calculate_roc_auc_score(y_true, predict_proba):
    return roc_auc_score(y_true, y_score=predict_proba)

def plot_roc_curve(clf, X_test, y_test):
    RocCurveDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig('roc_auc_curve.png')





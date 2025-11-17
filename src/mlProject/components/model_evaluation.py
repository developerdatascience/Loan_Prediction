import mlflow
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score
    )
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json
from urllib.parse import urlparse
import joblib
from src.mlProject import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        self.model = joblib.load(self.config.model_path)
    

    def _evaluate_model(self, y_test, y_pred) -> Dict[str, Any]:
        """Comprehensive model evaluation needed"""

        # y_pred_proba = self.model.predict_proba(X_test)[:, 1] # type: ignore

        accuracy = accuracy_score(y_test, y_pred) # type: ignore
        precision = precision_score(y_test, y_pred, average='weighted') # type: ignore
        recall = recall_score(y_test, y_pred, average='weighted') # type: ignore
        f1 = f1_score(y_test, y_pred, average='weighted') # type: ignore

        reports_path = Path(f'{self.config.root_dir}/reports')
        if not reports_path.exists():
            reports_path.mkdir(parents=True, exist_ok=True)

        logger.info("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(reports_path, "confusion_matrix.png"))
        plt.close()

        logger.info(f"Classification Report \n:{classification_report(y_test, y_pred)}") # type: ignore

        # logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        # roc_auc_value= roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Model evaluation metrics:")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            # "roc_auc_score": float(roc_auc_value)
        }

        return metrics
    
    def log_into_mlflow(self) -> None:
        test_data = pd.read_csv(self.config.test_data_path)

        text_x = test_data.drop(self.config.target_column, axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            prediction = self.model.predict(text_x)

            scores = self._evaluate_model(y_test=test_y, y_pred=prediction)
            # saving metrics as local
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(metrics=scores)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="XGBoost")
            else:
                mlflow.sklearn.log_model(self.model, "model")




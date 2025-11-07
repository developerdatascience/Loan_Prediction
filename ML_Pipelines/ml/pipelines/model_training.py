import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score
    )
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import shap
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelTrainingPipeline:
    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Initialize the ModelTrainingPipeline

        Args:
            data (pd.DataFrame): input dataset
            model (object): machine learning model
            target_column (str): name of the target column
        """
        self.data = data
        self.target_column = target_column
        self.model = None
        self.standard_scaler = StandardScaler()

    def divide_and_standardize_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        Returns:
            Tuple containing training features, testing features, training labels, and testing labels.
        """
        logger.info(f"Dividing data with test size = {test_size}")

        # Implementation for dividing data goes here
        X = self.data.drop(columns=self.target_column)
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        X_train = self.standard_scaler.fit_transform(X_train)
        X_test = self.standard_scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test # type: ignore


    def train_model(self,model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        Train various machine learning models on the provided dataset.
        """
        logger.info("Starting model training pipeline.")

        if model_name is None:
            logger.error("Model not provided.")
            raise ValueError("Model must be provided for training.")
        
        if model_name == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_name == "XGBoost":
            self.model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        elif model_name == "LightGBM":
            self.model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        else:
            logger.error(f"Unsupported model provided: {model_name}")
            raise ValueError(f"Unsupported model: {model_name}")

        # reassure static type checkers that self.model is set
        assert self.model is not None

        self.model.fit(X_train, y_train) 
        logger.info(f"{model_name} model trained successfully.")

        return self.model
    
    def evaluate_model(self, X_test, y_test) -> Dict[str, Any]:
        """Comprehensive model evaluation needed"""
        y_pred = self.model.predict(X_test) # type: ignore

        y_pred_proba = self.model.predict_proba(X_test)[:, 1] # type: ignore

        accuracy = accuracy_score(y_test, y_pred) # type: ignore
        precision = precision_score(y_test, y_pred, average='weighted') # type: ignore
        recall = recall_score(y_test, y_pred, average='weighted') # type: ignore
        f1 = f1_score(y_test, y_pred, average='weighted') # type: ignore

        reports_path = Path('reports')
        if not reports_path.exists():
            reports_path.mkdir(parents=True, exist_ok=True)

        logger.info("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('reports/confusion_matrix.png')
        plt.close()

        logger.info(f"Classification Report \n:{classification_report(y_test, y_pred)}") # type: ignore

        logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        roc_auc_value= roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Model evaluation metrics:")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc_score": float(roc_auc_value),
            "confusion_matrix": cm
        }
    
    def explain_prediction(self, X_sample, feature_names):
        """Generate model explainability

        Args:
            X_test (pd.DataFrame): The input features for the test set.

        Returns:
            pd.DataFrame: A DataFrame containing feature importances.
        """

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('reports/shap_summary_plot.png')
        plt.close()
        
        return shap_values



    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file.

        Args:
            filepath (str): The path where the model should be saved.
        Returns:
            None
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

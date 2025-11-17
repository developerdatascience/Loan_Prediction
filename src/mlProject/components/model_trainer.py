import logging
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import shap
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainingPipeline

        Args:
            config (ModelTrainerConfig): config
            model (object): machine learning model
            target_column (str): name of the target column
        """
        self.config = config
        self.standard_scaler = StandardScaler()
        self.model = None

    def divide_and_standardize_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        Returns:
            Tuple containing training features, testing features, training labels, and testing labels.
        """
        logger.info(f"Dividing data with test size = {self.config.test_size}")

        # Implementation for dividing data goes here
        data = pd.read_csv(self.config.train_data_path)
        X = data.drop(columns=self.config.target_column, axis=1)
        y = data[self.config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=42, stratify=y)
        
        X_train = self.standard_scaler.fit_transform(X_train)
        X_test = self.standard_scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test # type: ignore


    def train_model(self,) -> object:
        """
        Train various machine learning models on the provided dataset.
        """
        logger.info("Starting model training pipeline.")

        X_train, X_test, y_train, y_test = self.divide_and_standardize_data()

        if self.config.model_name is None:
            logger.error("Model name not provided.")
            raise ValueError("Model must be provided for training.")
        
        if self.config.model_name.lower() == "randomforest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.config.model_name.lower() == "xgboost":
            self.model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        elif self.config.model_name.lower() == "lightgbm":
            self.model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        else:
            logger.error(f"Unsupported model provided: {self.config.model_name}")
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        self.model.fit(X_train, y_train) 
        logger.info(f"{self.config.model_name} model trained successfully.")
        
        logger.info(f"{self.config.model_name} saved at {self.config.root_dir}")
        joblib.dump(self.model, os.path.join(self.config.root_dir, "model.joblib"))

        # reassure static type checkers that self.model is set
        # assert self.model is not None


        return self.model
    
    
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

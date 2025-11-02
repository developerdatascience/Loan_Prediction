import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class ModelTrainingPipeline:
    def __init__(self, data, target_column: str, model_name):
        """I
        Initialize the ModelTrainingPipeline

        Args:
            data (_type_): input dataset
            model (_type_): machine learning model
            target_column (str): name of the target column
        """
        self.data = data
        self.target_column = target_column
        self.model_name = model_name
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train = self.standard_scaler.fit_transform(X_train)
        X_test = self.standard_scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test # type: ignore
   
    def train_model(self) -> None:
        """
        Train various machine learning models on the provided dataset.
        """
        logger.info("Starting model training pipeline.")

        X_train, X_test, y_train, y_test = self.divide_and_standardize_data()

        if self.model_name is None:
            logger.error("Model not provided.")
            raise ValueError("Model must be provided for training.")
        
        if self.model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            logger.info(f"RandomForest model accuracy: {accuracy:.4f}")

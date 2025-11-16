import logging
import os
import pandas as pd
from src.mlProject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from typing import Callable, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self,
                 config: DataTransformationConfig,
                 transformation_steps: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> None:
        """
        Intialize the tranformation class

        Args:
            transformation_steps (List[Callable[[pd.DataFrame], pd.DataFrame]]): 
                A list of transformation functions to apply sequentially.
                Transformation steps include functions like:
                    - drop_columns_having_nulls_above_threshold
                    - fill_numeric_missing_values
                    - fill_categorical_missing_values
                    - fill_numeric_missing_values_using_interpolation
                steps = [drop_columns_having_nulls_above_threshold, 
                     fill_numeric_missing_values, 
                     fill_categorical_missing_values,
                     fill_numeric_missing_values_using_interpolation]
        """
        self.config = config
        self.transformation_steps = transformation_steps
        self.data = None

    def _run_transformation(self) -> pd.DataFrame:
        """Run the transformation pipeline."""

        self.data = pd.read_csv(self.config.data_path)

        if self.data is None:
            pass

        if self.transformation_steps is None or len(self.transformation_steps) == 0:
            logger.info("No transformation steps provided. Skipping transformation.")
            return self.data
        


        for step in self.transformation_steps:
            logger.info("=================================")
            logger.info(f"Applying transformation step: {step.__name__}")
            logger.info("=================================")
            self.data = step(self.data)
        
        logger.info("Data transformation completed.")

        return self.data
    
    def train_test_split(self) -> None:
        self.data = self._run_transformation()

        train, test = train_test_split(self.data)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted data into training and test data")
        logger.info(train.shape)
        logger.info(test.shape)

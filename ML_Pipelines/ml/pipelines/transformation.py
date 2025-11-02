import logging
import pandas as pd
from ML_Pipelines.ml.pipelines.ingestion import IngestionPipeline
from ML_Pipelines.ml.utils.utility import (
    drop_columns_having_nulls_above_threshold,
    fill_numeric_missing_values,
    fill_categorical_missing_values,
    fill_numeric_missing_values_using_interpolation
)
from typing import Callable, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformationPipeline:
    def __init__(self, 
                 ingestion_pipeline: IngestionPipeline, 
                 transformation_steps: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> None:
        """
        Intialize the tranformation pipeline

        Args:
            ingestion_pipeline (IngestionPipeline): The ingestion pipeline instance.
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
        self.ingestion_pipeline = ingestion_pipeline
        self.transformation_steps = transformation_steps
        self.data = None

    def run_transformation(self) -> pd.DataFrame:
        """Run the transformation pipeline."""
        if self.data is None:
            self.data = self.ingestion_pipeline.load_data()

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
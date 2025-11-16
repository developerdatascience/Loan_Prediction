from src.mlProject.config.configuration import ConfiguratonManager
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject.utils.utility import (
    drop_columns_having_nulls_above_threshold,
    fill_numeric_missing_values,
    fill_categorical_missing_values,
    fill_numeric_missing_values_using_interpolation,
    drop_columns_with_IDs,
    encode_cat_columns
)
from src.mlProject import logger


STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfiguratonManager()
        data_transformation_config = config.get_data_transformation_config()
        steps = [drop_columns_with_IDs,
         drop_columns_having_nulls_above_threshold, 
         fill_numeric_missing_values, 
         fill_categorical_missing_values,
        encode_cat_columns]
        data_tranformation = DataTransformation(config=data_transformation_config,
                                                transformation_steps=steps)
        data_tranformation.train_test_split()
    

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
        


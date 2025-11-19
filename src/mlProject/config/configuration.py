from src.mlProject.constants import *
from src.mlProject.utils.common import create_directories, read_yaml
from src.mlProject.entity.config_entity import (DataIngesitonConfig, 
                                                  DataTransformationConfig,
                                                  DataValidationConfig,
                                                  ModelTrainerConfig,
                                                  ModelEvaluationConfig)


class ConfiguratonManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
    
    def get_data_ingestion_config(self) -> DataIngesitonConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngesitonConfig(
            root_dir=config.root_dir,
            source_URL= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

         
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir= config.root_dir,
            all_schema=schema,
            STATUS_FILE= config.STATUS_FILE,
            unzip_data_dir= config.unzip_data_dir
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        target_column = self.schema.TARGET_COLUMN.name

        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir= config.root_dir,
            target_column=target_column,
            data_path= config.data_path
        )


    def get_data_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        target_column = self.schema.TARGET_COLUMN.name
        alpha = self.params.ElasticNet.alpha
        l1_ratio = self.params.ElasticNet.l1_ratio

        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path= config.test_data_path,
            model_name= config.model_name,
            test_size=config.test_size,
            alpha=alpha,
            l1_ratio=l1_ratio,
            target_column=target_column
        )

    def get_data_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        all_params = self.params.ElasticNet
        target_column = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=all_params,
            target_column=target_column.name,
            metric_file_name=config.metric_file_name,
            mlflow_uri="http://127.0.0.1:5000"
            )

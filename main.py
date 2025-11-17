from src.mlProject import logger
from src.mlProject.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from src.mlProject.pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from src.mlProject.pipelines.stage_03_data_transformation import DataTransformationPipeline
from src.mlProject.pipelines.stage_04_model_trainer import ModelTrainerPipeline
from src.mlProject.pipelines.stage_05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stages"

try:
    logger.info(f">>>>>>>>>>>>>>{STAGE_NAME} started<<<<<<<<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>>>>>>>>>{STAGE_NAME} started<<<<<<<<<<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e
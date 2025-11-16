from src.mlProject.config.configuration import ConfiguratonManager
from src.mlProject.components.data_validation import DataValidation
from src.mlProject import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguratonManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

        

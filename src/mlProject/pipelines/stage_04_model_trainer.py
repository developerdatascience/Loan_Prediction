from src.mlProject.config.configuration import ConfiguratonManager
from src.mlProject.components.model_trainer import DataModelTrainer
from src.mlProject import logger

STAGE_NAME = "Model Trainer"

class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfiguratonManager()
        model_trainer_config = config.get_data_model_trainer_config()
        model = DataModelTrainer(config=model_trainer_config)
        model.train_model()

if __name__ == "__main__":
    try:
        logger.info(">>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(">>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

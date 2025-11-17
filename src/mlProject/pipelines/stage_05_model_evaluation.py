from src.mlProject import logger
from src.mlProject.components.model_evaluation import ModelEvaluation
from src.mlProject.config.configuration import ConfiguratonManager

STAGE_NAME = "Model Evaluation Stage"
class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfiguratonManager()
        model_eval_config = config.get_data_model_evaluation_config()
        model_eval = ModelEvaluation(config=model_eval_config)
        model_eval.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage started<<<<<<<<<<<<<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>>>>>>>>>>{STAGE_NAME} stage completed<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

import os
import logging
import pandas as pd
from pathlib import Path
from ML_Pipelines.ml.utils.data_loader import get_latest_partition_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IngestionPipeline:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.filename = get_latest_partition_data(data_dir=self.data_dir)

        if not Path(self.data_dir).exists():
            logger.info(f"{data_dir} does not exists.!!")

    def load_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, self.filename)

        if not Path(file_path).exists():
            logger.error(f"{file_path} not found.")
            raise FileNotFoundError(f"{file_path} not found.")

        file_extension = file_path.split("/")[-1].split(".")[-1]

        try:
            if file_extension == "csv":
                data = pd.read_csv(file_path).dropna(how='all')
                logger.info(f"Latest partition file {file_path} loaded successfully.")
            elif file_extension in ("xlsx", "xls"):
                data = pd.read_excel(file_path).dropna(how='all')
                logger.info(f"Latest partition file {file_path} loaded successfully.")
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise ValueError(f"Unsupported file type: {file_extension}")

            # At this point data is guaranteed to be a DataFrame; return it.
            return data
        except Exception:
            logger.exception(f"Error loading {file_path}")
            raise
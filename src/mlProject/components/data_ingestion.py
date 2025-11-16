import os
from datetime import datetime
import logging
import pandas as pd
from pathlib import Path
from src.mlProject.utils.data_loader import get_latest_partition_data
from src.mlProject.entity.config_entity import DataIngesitonConfig
from src.mlProject.utils.common import get_size
import kagglehub
import shutil
from src.mlProject import logger



class DataIngestion:
    def __init__(self, config: DataIngesitonConfig) -> None:
        self.config = config
    
    def download_file(self):
        path = kagglehub.dataset_download(self.config.source_URL)
        if os.path.exists(self.config.local_data_file):
            shutil.copy(self.config.local_data_file, self.config.unzip_dir)
            logger.info(f"File copied to local directory-{self.config.unzip_dir}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.unzip_dir))}")
    

    def extract_file(self):
        raise NotImplementedError





# class IngestionPipeline:
#     def __init__(self, data_dir: Path) -> None:
#         self.data_dir = data_dir
#         self.filename = get_latest_partition_data(data_dir=self.data_dir)
#         self.current_date = datetime.now().strftime("%Y%m%d")

#         if not Path(self.data_dir).exists():
#             logger.info(f"{data_dir} does not exists.!!")

#     def load_data(self) -> pd.DataFrame:
#         file_path = os.path.join(self.data_dir, self.filename)

#         if not Path(file_path).exists():
#             logger.error(f"{file_path} not found.")
#             raise FileNotFoundError(f"{file_path} not found.")

#         file_extension = file_path.split("/")[-1].split(".")[-1]

#         try:
#             if file_extension == "csv":
#                 data = pd.read_csv(file_path).dropna(how='all')
#                 logger.info(f"Latest partition file {file_path} loaded successfully.")

#                 data.to_csv(f"data_folder/train/credit_train_{self.current_date}.csv", index=False)
#             elif file_extension in ("xlsx", "xls"):
#                 data = pd.read_excel(file_path).dropna(how='all')
#                 logger.info(f"Latest partition file {file_path} loaded successfully.")
#                 data.to_excel(f"data_folder/train/credit_train_{self.current_date}.xlsx", index=False)
#             else:
#                 logger.error(f"Unsupported file type: {file_extension}")
#                 raise ValueError(f"Unsupported file type: {file_extension}")

#             # At this point data is guaranteed to be a DataFrame; return it.
#             return data
#         except Exception:
#             logger.exception(f"Error loading {file_path}")
#             raise
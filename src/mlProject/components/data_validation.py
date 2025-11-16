import pandas as pd
from src.mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    
    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            data = pd.read_csv(self.config.unzip_data_dir)
            all_columns = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_columns:
                validation_status = col in all_schema
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e

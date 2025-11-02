import os

def create_file(file_path, content=""):
    """Create a file with the given content."""
    with open(file_path, 'w') as file:
        file.write(content)


def create_project_directory(base_path="ML_Pipelines"):
    """Create a project directory structure"""
    structure = [
     "ml/core/__init__.py",
     "ml/core/config.py",
     "ml/core/database.py",
     "ml/core/security.py",
     "ml/models/__init__.py",
     "ml/pipelines/__init__.py",
     "ml/pipelines/ingestion.py",
     "ml/pipelines/transformation.py",
     "ml/pipelines/model_training.py",
     "ml/pipelines/model_testing.py",
     "ml/pipelines/evaluation.py",
     "ml/utils/__init__.py",
     "ml/utils/data_loader.py",
     "ml/utils/metrics.py",
     "ml/mlops/__init__.py",
     "ml/notebooks/IngestionPipeline.ipynb",
     "ml/notebooks/TransformationPipeline.ipynb"
    ]

    for file_path in structure:
        full_path = os.path.join(base_path, file_path)
        dir_name = os.path.dirname(full_path)
        os.makedirs(dir_name, exist_ok=True)
        create_file(full_path)
    
    print(f"Project directory structure created under '{base_path}'")


if __name__ == "__main__":
    create_project_directory()
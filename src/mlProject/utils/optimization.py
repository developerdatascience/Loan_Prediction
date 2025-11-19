from sklearn.model_selection import RandomizedSearchCV
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from src.mlProject import logger


class HyperparameterTuning:
    def __init__(self, 
                 n_estimators: List[int], 
                 max_features: List[int], 
                 max_depth: List[int], 
                 min_samples_split: List[int],
                 min_samples_leaf: List[int],
                 bootstrap: List[bool]) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
    
    def optimize(self, X_train, y_train) -> Dict[str, Any]:
        random_grid = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "bootstrap": self.bootstrap
        }
        classifer = RandomForestClassifier()

        model_tuning = RandomizedSearchCV(estimator=classifer,
                                          param_distributions=random_grid,
                                          n_iter=10,
                                          cv=5,
                                          random_state=42,
                                          verbose=2,
                                          n_jobs=-1)
        
        model_tuning.fit(X_train, y_train)
        logger.info(f"Random Grid: {random_grid}")

        logger.info(f"best_params: {model_tuning.best_params_}")

        best_params = model_tuning.best_params_

        return best_params



        
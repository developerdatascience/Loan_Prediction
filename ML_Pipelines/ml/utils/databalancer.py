import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class DataBalancer:
    """
    Detects and balances imbalanced datasets automatically
    """
    def __init__(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Initialize the DataBalancer with dataframe and target_column

        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
        """
        self.data = data.copy()
        self.target_column = target_column
        self.balance_ratio = None
    
    def detect_imbalance(self, plot: bool = True):
        """Detect the imbalance in the target column and optionally plot class distribution

        Args:
            plot (bool, optional): Whether to plot class distribution. Defaults to True.
        """
        counts = self.data[self.target_column].value_counts()
        self.balance_ratio = counts.min() / counts.max()

        if plot:
            plt.figure(figsize=(8, 6))
            sns.barplot(x=counts.index.astype(str), y=counts.values)
            plt.title(f"Class Distribution for '{self.target_column}")
            plt.xlabel("Class")
            plt.ylabel("Frequency")
            plt.show()
    

    def choose_method(self) -> str:
        """choose balancing method based on imbalance severity

        Returns:
            str: give best balancing method
        """
        if self.balance_ratio is None:
            self.detect_imbalance(plot=False)
        
        if self.balance_ratio is None or self.balance_ratio >=0.8:
            method = 'None'
        elif self.balance_ratio >=0.5:
            method = 'over'
        elif self.balance_ratio >= 0.3:
            method = 'smote'
        else:
            method = "under" if len(self.data) > 1000 else "smote"

        print(f"🤖 Auto-selected method based on imbalance ratio ({self.balance_ratio:.2f}): {method.upper()}")

        return method

    def balance_data(self, method: str = None, random_state: int = 42) -> pd.DataFrame:
        """
        Balances the dataset using the specified or automatically chosen method.

        Args:
            method (str, optional): One of ['smote', 'over', 'under', 'none']
            random_state (int): Random seed for reproducibility

        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        if method is None or method.lower() == 'auto':
            method = self.choose_method()
        
        if method.lower() == "none":
            print("Data already balance. No resampling needed.")
            return self.data
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        if method.lower() == 'smote':
            balancer = SMOTE(random_state=random_state)
        elif method.lower() == 'over':
            balancer = RandomOverSampler(random_state=random_state)
        elif method.lower() == 'under':
            balancer = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError("❌ method must be one of ['smote', 'over', 'under', 'none']")
        
        X_res, y_res = balancer.fit_resample(X, y)
        balanced_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), 
                                 pd.Series(y_res, name=self.target_column)], axis=1)

        print(f"✅ Balancing done using: {method.upper()}")
        print("New class distribution:")
        print(Counter(y_res))

        return balanced_df
    
    @staticmethod
    def visualize_comparison(original_df, balanced_df, target_column: str) -> None:
        """
        Compare original vs balanced class distributions visually.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(x= target_column, data=original_df, ax=axes[0], palette='coolwarm')

        axes[0].set_title("Before Balancing")
        axes[0].set_ylabel("Count")

        sns.countplot(x=target_column, data=balanced_df, ax=axes[1], palette="coolwarm")
        axes[1].set_title("After Balancing")

        plt.tight_layout()
        plt.show()
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import warnings
warnings.filterwarnings("ignore")


class OutlierTreatment:
    def __init__(self, df: pd.DataFrame, treatment_type: str, numerical_features: List[str], target_col) -> None:
        self.df = df
        self.treat_type = treatment_type
        self.num_feats = numerical_features
        self.target_col = target_col


    def _plot_features(self):
        for c in self.num_feats:
            plt.figure(figsize=(16,5))
            plt.subplot(1,2,1)
            fig = sns.distplot(self.df[self.target_col])
            plt.savefig("img/distribution.png", bbox_inches="tight")
            
        



    def _z_score(self) -> pd.DataFrame:
        pass
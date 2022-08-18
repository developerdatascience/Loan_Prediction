from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import warnings
warnings.filterwarnings("ignore")


class OutlierTreatment:
    def __init__(self, 
                df: pd.DataFrame, 
                treatment_type: str, 
                numerical_features: List[str], 
                output_type : str,
                plot:bool=  False) -> None:
        """
        df : pandas Dataframe
        treatment_type: [z_score, iqr, winsorization]
        numerical_features: List of numerical columns
        output_type: Trim or Capping
        Trim: This will give a subset of original dataframe with all outlier removed
        Capping: Incase of outlier, it will cap the outlier value with high and low values

        Winsorization : While we remove the outliers using capping, then that particular method is known as Winsorization.
        """

        self.df = df
        self.treat_type = treatment_type
        self.num_feats = numerical_features
        self.plot = plot
        self.output_type = output_type
        self.output_df = self.df.copy(deep=True)


    def __plot_features(self):
        for c in self.num_feats:
            plt.figure(figsize=(16,5))
            plt.subplot(1,2,1)
            sns.boxplot(self.df[c])
            plt.savefig(f"img/{c}.png", bbox_inches="tight")

    def __z_score(self) -> pd.DataFrame:
        for c in self.num_feats:
            ho = self.df[c].mean() + 3 * self.df[c].std()
            lo = self.df[c].mean() - 3 * self.df[c].std()
            print("Highest allowed", ho)
            print("Lowest allowed", lo)
            if self.output_type == "Trim":
                self.output_df = self.df[(self.df[c] < ho) & (self.df[c] > lo)]
                return self.output_df
            elif self.output_type == "Capping":
                self.df[c] = np.where(self.df[c] > ho, ho, 
                                    np.where(self.df[c] < lo, lo, self.df[c]))
                return self.df
            else:
                raise Exception("Output type not understood")
    
    def __iqr(self) -> pd.DataFrame:
        for c in self.num_feats:
            percentile25 = self.df[c].quantile(0.25)
            percentile75 = self.df[c].quantile(0.75)

            upper_limit = percentile75 + 1.5 * 0.1
            lower_limit = percentile25 + 1.5 * 0.1

            if self.output_type == "Trim":
                self.output_df = self.df[self.df[c] < upper_limit]
                return self.output_df
            elif self.output_type == "Capping":
                self.output_df = np.where(self.df[c] >  upper_limit, upper_limit,
                                        np.where(self.df[c] < lower_limit, lower_limit, self.df[c]))
                return self.output_df
            else:
                raise Exception("Output type not understood")
    
    def __winsorization(self) -> pd.DataFrame:
        for c in self.num_feats:
            upper_limit = self.df[c].quantile(0.99)
            lower_limit = self.df[c].quantile(0.01)

            if self.output_type == "Trim":
                self.output_df = self.df[(self.df[c] <= upper_limit) & (self.df[c] >= lower_limit)]
                return self.output_df
            elif self.output_type == "Capping":
                self.df[c] = np.where(self.df[c] >= upper_limit, upper_limit,
                                np.where(self.df[c] <= lower_limit, lower_limit, self.df[c]))
            else:
                raise Exception("Output type not understood")


    def fit_transform(self):
        if self.plot:
            return self.__plot_features()
        print("Plotting Completed...")
                    
        if self.treat_type == "z_score":
            return self.__z_score()
        elif self.treat_type == "iqr":
            return self.__iqr()
        elif self.treat_type == "winsorization":
            return self.__winsorization()
        
    
    

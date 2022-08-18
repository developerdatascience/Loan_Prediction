from statistics import mode
import pandas as pd
import numpy as np
from typing import List, Union


class Exploratory:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self._missing_col: List[str] = self.df.columns[self.df.isnull().any()]


    def missing_na(self) -> pd.DataFrame:
        for c in self._missing_col:
            if c in list(set(self.df.columns[self.df.dtypes=="object"])):
                self.df[c] = self.df.loc[:, c].astype(str).fillna(self.df.loc[:, c].mode())
            elif c in list(set(self.df.columns[(self.df.dtypes=="float")|(self.df.dtypes=="int")])):
                self.df[c] = self.df.loc[:, c].fillna(self.df.loc[:, c].mean())
            else:
                raise Exception("Something went wrong!!")
        print(self.df.isnull().sum())
        return self.df
    

    def outlier_detection(self) -> List[Union[int, float]]:
        outlier = List[Union[int, float]]
        for c in self.df.columns[(self.df.dtypes == "float") | (self.df.dtypes == "int")]:
            col_mean = self.df[c].mean()
            std_dev = np.std(self.df[c])
            z = (self.df[c] - col_mean)/std_dev
            if (z > col_mean).any():
                outlier.append(c)
            print("mean of {} is {} and standard Deviation is {} ".format(c, col_mean,std_dev))    
            print(f"outlier in {c}")
        return outlier

    
    
        


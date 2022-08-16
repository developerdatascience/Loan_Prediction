from statistics import mode
import pandas as pd
from typing import List


class Exploratory:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.missing_col: List[str] = self.df.columns[self.df.isnull().any()]


    def missing_na(self) -> pd.DataFrame:
        for c in self.missing_col:
            if c in list(set(self.df.columns[self.df.dtypes=="object"])):
                self.df[c] = self.df.loc[:, c].astype(str).fillna(self.df[:, c].mode())
            elif c in list(set(self.df.columns[(self.df.dtypes=="float")|(self.df.dtypes=="int")])):
                self.df[c] = self.df.loc[:, c].fillna(self.df.loc[:, c].mean())
            else:
                raise Exception("Something went wrong!!")
        return self.df
    

    def outlier_detection(self):
        pass
    
        


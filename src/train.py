import pandas as pd
import os
from utils import Exploratory

from sklearn import preprocessing


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))


FOLD_MAPPING = {
        0: [1, 2, 3, 4, 5],
        1: [0, 2, 3, 4, 5],
        2: [0, 1, 3, 4, 5],
        3: [0, 1, 2, 4, 5],
        4: [0, 1, 2, 3, 5],
        5: [0, 1, 3, 3, 4]
        }

if __name__ == "__main__":
        df = pd.read_csv(TRAINING_DATA)
        df_test = pd.read_csv(TEST_DATA)

        df_train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        df_valid = df[df.kfold == FOLD].reset_index(drop=True)

        df_train = df_train.drop(["Loan_ID","Loan_Status"], axis=1)
        df_valid = df_valid.drop(["Loan_ID", "Loan_Status"], axis=1)

        ytrain = df_train["Loan_Status"].values
        yvalid = df_valid["Loan_Status"].values

        df_valid = df_valid[df_train.columns]

        cat_feats = df_train.columns[df_train.dtypes=="object"]
        
        exp = Exploratory(dataframe=df_train)
        exp.missing_na()

        label_encoder = dict()
        for c in cat_feats:
                lbl = preprocessing.LabelEncoder()
                df_train.loc[:, c] = df_train.loc[:, c].astype(str).fillna("NONE")
                df_valid.loc[:, c] = df_valid.loc[:, c].astype(str).fillna("NONE")
                df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
                lbl.fit(df_train[c].values.to_list()+
                        df_valid[c].values.to_list()+
                        df_test[c].values.to_list()
                )
                df_train.loc[:, c] = lbl.fit_transform(df_train[c].values)
                df_test.loc[:, c] = lbl.fit_transform(df_test[c].values)
                label_encoder[c] = lbl
        
        


        

        





import pandas as pd
import numpy as np
import utils
from outlier import OutlierTreatment

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    
    utils.Exploratory(dataframe=df).missing_na()

    df = df.drop("Loan_ID", axis=1)

    utils.Exploratory(dataframe=df).outlier_detection()

    # Outlier Treament
    num_feats = df.columns[(df.dtypes == "float") | (df.dtypes == "int")]

    treat = OutlierTreatment(df=df, 
                        treatment_type="z_score", 
                        numerical_features=num_feats, 
                        output_type= "Capping",
                        plot=True)

    treat.fit_transform()

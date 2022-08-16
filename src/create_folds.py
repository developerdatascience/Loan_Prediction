import enum
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X = df, y=df["Loan_Status"].values)):
        print("train: ", len(train_idx), "valid: ", len(valid_idx))
        df.loc[valid_idx, "kfold"] = fold
    
    df.to_csv("input/train_folds.csv", index=False)
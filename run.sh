export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

FOLD=$1 
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train

python3 -m src.train

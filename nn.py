import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pickle
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 10
CSV_FILE = 'StackOverflowData.csv'
OUTPUT_FILE = 'nn.obj'

def nn():
    df = pd.read_csv(CSV_FILE)
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]
    NFEATURES = len(X_df.iloc[0])
    nn = MLPClassifier(solver='lbfgs', alpha='1e-4', hidden_layer_sizes=(NFEATURES, 2), random_state=RANDOM_STATE)
    eval = cross_validate(nn, X_df, y_df, cv=4)
    print(eval)

def main():
    if not os.path.exists(OUTPUT_FILE):
        nn()

if __name__=='__main__':
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

CSV_FILE = os.path.join('./dataset','train.csv')
OUTPUT_FOLDER = './Outputs'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'log_reg_results.obj')
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'log_reg_model.obj')

def log_reg():
    df = pd.read_csv(CSV_FILE)
    df = df.reset_index()
    # df.dropna(inplace=True)
    df = df[ ~df.isin([np.nan, np.inf, -np.inf]).any(1) ]
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]
    # np.where(X_df.values >= np.finfo(np.float64).max)
    # print(len(df))
    log_reg = LogisticRegressionCV(cv=5, random_state=10).fit(X_df, y_df) #Cs=4, fit_intercept=True, cv=10, verbose =1, random_state=42)
    
    # print(X_df.shape, y_df.shape)
    # with cross validation
    evals = cross_validate(log_reg, X_df, y_df, cv=5)
    print(evals)
    coeffs = log_reg.coef_
    scores = log_reg.scores_
    print(len(i) for i in scores)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(coeffs, f)
    f.close()
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(log_reg, f)
    f.close()

def main():
    start_time = time.time()
    log_reg()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()

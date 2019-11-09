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
# OUTPUT_FILE = os.path.join('./Outputs','log_reg.obj')
OUTPUT_FOLDER = './Outputs'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'log_reg_results.obj')
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'log_reg_model.obj')

def log_reg():
    df = pd.read_csv(CSV_FILE)
    print(df.head())
    df.dropna(inplace=True)
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]
    # print(X_df.head())
    # print(y_df.head())
    log_reg = LogisticRegressionCV(cv=5, random_state=10).fit(X_df, y_df) #Cs=4, fit_intercept=True, cv=10, verbose =1, random_state=42)

    print(X_df.shape, y_df.shape)
    # with cross validation
    evals = cross_validate(log_reg, X_df, y_df, cv=5)
    print(evals)
    coeffs = log_reg.coef_
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

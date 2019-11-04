import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

CSV_FILE = 'StackOverflowData.csv'
OUTPUT_FILE = 'log_reg.obj'

def log_reg():
    df = pd.read_csv(CSV_FILE)
    # print(df.head())
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]

    log_reg = LogisticRegressionCV(cv=5, random_state=10).fit(X_df, y_df) #Cs=4, fit_intercept=True, cv=10, verbose =1, random_state=42)

    print(X_df.shape, y_df.shape)
    # with cross validation
    evals = cross_validate(log_reg, X_df, y_df, cv=5)
    print(evals)
    coeffs = log_reg.coef_
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(coeffs, f)

def main():
    if not os.path.exists(OUTPUT_FILE):
        log_reg()

if __name__==__main__:
    main()

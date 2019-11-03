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

    log_reg = LogisticRegression()

    # without cross validation, requires change in order
    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=10)
    # log_reg.fit(X_train, y_train)
    # pred = log_reg.predict(X_test)
    # print(classification_report(y_test, pred))
    # coeffs = log_reg.coef_

    # with cross validation
    evals = cross_validate(log_reg, X_df, y_df, cv=3)
    print(evals)
    pickle.dump(coeffs, OUTPUT_FILE)

if __name__==__main__:
    if not os.path.exists(OUTPUT_FILE):
        log_reg()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm
import os
import pickle
import warnings
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
CSV_FILE = os.path.join('./dataset','train.csv')
OUTPUT_FOLDER = './Outputs'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'svm_results.obj')
MODEL_FILE = os.path.join(OUTPUT_FOLDER, 'svm_model.obj')

#X_df = df
# X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.33, random_state=10)

def svm_reg():
    df = pd.read_csv(CSV_FILE)
    # print(df.head())
    
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]

    lsvm = svm.SVC(kernel='linear',C = 1.0, random_state=10).fit(X_df, y_df) #Cs=4, fit_intercept=True, cv=10, verbose =1, random_state=42)
    
    
    print(X_df.shape, y_df.shape)
    #print(summary(lsvm))
    # with cross validation
    evals = cross_val_score(lsvm, X_df, y_df, cv=5)
    
    #print(evals)
    #coeffs = log_reg.coef_
    #with open(OUTPUT_FILE, 'wb') as f:
        #pickle.dump(coeffs, f)
        

def main():
    if not os.path.exists(OUTPUT_FILE):
        svm_reg()

if __name__=='__main__':
    main() 
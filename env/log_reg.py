import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

CSV_FILE = 'StackOverflowData.csv'

df = pd.read_csv(CSV_FILE)
print(df.head())
train_df = df.iloc[:,:-1]
test_df = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.3, random_state=10))

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print(classification_report(y_test, pred))
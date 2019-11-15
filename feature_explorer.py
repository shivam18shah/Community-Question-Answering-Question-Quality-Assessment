import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import os
from scipy.fftpack import fft

FILE_PATH = os.path.join('dataset', 'train.csv')
df = pd.read_csv(FILE_PATH)

x = df.iloc[:,4]
y = fft(x)

plt.plot(x, y)
plt.show()
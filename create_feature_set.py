%pip install pycaret

from pycaret.regression import setup , compare_models , create_model , tune_model , plot_model , predict_model , finalize_model , save_model , load_model, evaluate_modelimport numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def call_data():
    df = pd.read_csv("MES 06-21.Last.txt", names=['time', 'open', 'high', 'low', 'close', 'volume'], delimiter = ";",)
    df.time = pd.to_datetime(df.time, format = '%Y.%m.%d %H:%M:%S.%f')
    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time\ 
    df = df.drop_duplicates()
    return df
# read csv
data = pd.read_csv('MES 06-21 Indicators.csv', index_col='time', parse_dates=True)
data['Date'] = pd.to_datetime(data.index)
data.head()

s = setup(data,target = 'close')
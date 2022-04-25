import numpy as np
import pandas as pd

train_df = pd.read_csv("../dataset/temp.csv",header=0)
train_data = train_df.values
print(train_data[:,1:])
print(train_data[:,0])
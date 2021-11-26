import pandas as pd
import matplotlib
data=pd.read_csv("healthcare-dataset-stroke-data.csv")
# data.head()
data=data.drop(columns=["id"],axis=1)
data.shape
# (5110,11)
data.isnull().sum()
data.describe()
data.info()
data.stroke.value_counts()
# 0    4861
# 1     249
data["bmi"].mean
#3.66
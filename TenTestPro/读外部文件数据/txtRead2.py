import numpy as np
import pandas as pd

# fileData = pd.read_csv('data.csv', dtype=np.float32, header=None, usecols=(1, 2, 3, 4))
# wholeData = fileData.values.tolist() # 将csv文件转为二维数组
# print(wholeData)

fileData = pd.read_csv('data.csv', dtype=np.float32, header=None, converters={(3): lambda s: 1.0 if s == "是" else 0.0})
wholeData = fileData.values.tolist() # 将csv文件转为二维数组
print(wholeData)
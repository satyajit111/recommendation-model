import sklearn
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
df1 = pd.read_csv(r"C:\Users\baps\Desktop\VEROFAX\ram.csv")

grouped=df1.groupby("Product Name")["Product Name"].agg('count')

print(grouped)

dfu=df1.drop_duplicates("Product Name")

grouped2=dfu.groupby("Product Name")["Product Name"].agg('count')

print(grouped2)

print(dfu)

dfu.to_csv('verofaxdataclean.csv',index=False)
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'realestate.csv')

print(f'The CSV file is saved at: {csv_path}')
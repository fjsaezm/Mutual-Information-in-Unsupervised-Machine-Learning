from os import listdir
from os.path import isdir
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_json(r"results.json")
print(df)

ax = sns.barplot(x="color_jitter",y = "label_top_1_accuracy",hue = "batch-size",data = df)
plt.show()
from os import listdir
from os.path import isdir
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_json(r"results.json")
df["regularization_loss"] = df["regularization_loss"].round(4)
df["label_top_1_accuracy"] = df["label_top_1_accuracy"].round(3)
df["label_top_5_accuracy"] = df["label_top_5_accuracy"].round(3)
df.round(3)
df.to_csv(path_or_buf = "results.csv")
#print(df)

#ax = sns.barplot(x="color_jitter",y = "label_top_1_accuracy",hue = "batch-size",data = df)
#plt.show()
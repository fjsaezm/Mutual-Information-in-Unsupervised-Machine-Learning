from os import listdir
from os.path import isdir
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_json(r"results-batch-comparison.json")
#df["regularization_loss"] = df["regularization_loss"].round(4)
df["label_top_1_accuracy"] = df["label_top_1_accuracy"].round(3)
df["label_top_5_accuracy"] = df["label_top_5_accuracy"].round(3)
df.round(3)
#df.to_csv(path_or_buf = "results.csv")
#print(df)

ax = sns.pointplot(x="batch-size",y = "label_top_1_accuracy",data = df,color = "green")


# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.savefig("../media/simclr-batch-comparison.pdf")
plt.show()
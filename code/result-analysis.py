from os import listdir
from os.path import isdir
import json
import pandas as pd
import seaborn as sns


df = pd.read_json(r"results.json")
print(df)

ax = sns.barplot(x="")
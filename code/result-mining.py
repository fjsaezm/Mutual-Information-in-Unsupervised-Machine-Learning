from os import listdir
from os.path import isdir
import json
import pandas as pd



# Path to look for models
path = "SimCLR/models/"
# Obtain all the directories of the models
dirs = [d for d in listdir(path) if isdir(path+d)]
dirs = [d for d in dirs if d.endswith('-50')]
dirs.sort(reverse=True)
index = [i for i in range(len(dirs))]
print(dirs)

# Create columns for the dataframe and empty dataframe
cols = ["batch_size","temperature","weight_decay","color_jitter","resnet_depth",
        "regularization_loss","label_top_1_accuracy","label_top_5_accuracy","global_step"]
df = pd.DataFrame(columns = cols)


for dir in dirs:
    path_json = path + dir + "/result.json"
    try:
        with open(path_json) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
    except:
        continue
    
    col = dir.split("-")
    for el in jsonObject:
        col.append(jsonObject[el])
        
    df.loc[-1] = col 
    df.index = df.index +1
    #df.sort_index()


df['batch_size'] = pd.to_numeric(df['batch_size'])
df = df.sort_values(by=["batch_size","temperature"])

df["regularization_loss"] = df["regularization_loss"].round(4)
df["label_top_1_accuracy"] = df["label_top_1_accuracy"].round(3)
df["label_top_5_accuracy"] = df["label_top_5_accuracy"].round(3)
df.round(3)
print(df)
df.to_csv(path_or_buf='results-simclr-resnet50.csv')
res = df.to_json("results-simclr-resnet50.json")

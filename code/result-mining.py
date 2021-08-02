from os import listdir
from os.path import isdir
import json
import pandas as pd



# Path to look for models
path = "SimCLR/models/"
# Obtain all the directories of the models
dirs = [d for d in listdir(path) if isdir(path+d)]
index = [i for i in range(len(dirs))]
print(dirs)

# Create columns for the dataframe and empty dataframe
cols = ["batch-size","temperature","weight_decay","color_jitter","resnet_depth",
        "eval/regularization_loss","eval/label_top_1_accuracy","eval/label_top_5_accuracy","global_step"]
df = pd.DataFrame(columns = cols)


for dir in dirs:
    path_json = path + dir + "/result.json"
    with open(path_json) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    
    col = dir.split("-")
    for el in jsonObject:
        col.append(jsonObject[el])
        
    df.loc[-1] = col 
    df.index = df.index +1
    df.sort_index()


print(df)
res = df.to_json(orient="split")
with open("results.json",'w') as outfile:
  json.dump(res,outfile)
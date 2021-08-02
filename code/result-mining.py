from os import listdir
from os.path import isdir
import json
import pandas as pd



# Path to look for models
path = "simclr/models/"
# Obtain all the directories of the models
dirs = [d for d in listdir(path) if isdir(join(path,d))]
index = [i for i in range(len(dirs))]
print(dirs)

# Create columns for the dataframe and empty dataframe
cols = ["batch-size","temperature","weight_decay","color_jitter","resnet_depth",
        "eval/regularization_loss","eval/label_top_1_accuracy","eval/label_top_5_accuracy","global_step"]
df = pd.DataFrame(index = index,columns = cols)


for dir in dirs:
    path_json = path + dir + "saved_model/result.json"
    with open(path_json) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    
    col = dir.split("-")
    for el in jsonObject:
        col.append(el)

    print(col)

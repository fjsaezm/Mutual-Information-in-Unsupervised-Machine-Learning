from tqdm import tqdm
import os
import itertools
from operator import add
import subprocess

COMMAND = """
python simclr/tf2/run.py --mode=train_then_eval  \
--train_epochs=100 --learning_rate=1.0   \
--dataset=cifar10 --image_size=32 --eval_split=test  \
--use_blur=False   \
--use_tpu=False
"""


"--model_dir=simclr/tf2/models/bs64",


headers = ["--train_batch_size=",
            "--temperature=",
            "--weight_decay=",
            "--color_jitter_strength=",
            "--resnet_depth=",
            "--model_dir="]

batch_sizes = [256]
temperatures = [0.5,0.75,1]
weight_decays = [1e-4]
color_jitter_strenghts = [0.5,0.75,1]
resnet_depths = [18,50]

all = [batch_sizes,temperatures,weight_decays,color_jitter_strenghts,resnet_depths]

for el in tqdm(list(itertools.product(*all))):
    # Create model dir name and create dir for the model
    model_dir =  '-'.join([str(elem) for elem in el])
    #os.mkdir(model_dir)
    # Join model dir name to the list of parameters
    list_el = [str(i) for i in el]
    list_el.append(model_dir)
    # Create the list of parameters
    to_add = list(map(add,headers,list_el))
    # Create a string with the command that will be executed
    # In this iteration
    COMMAND_ITERATION = COMMAND + ' '.join(to_add)
    #print(COMMAND_ITERATION)
    #os.system(COMMAND_ITERATION)


    

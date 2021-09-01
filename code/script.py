import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tqdm import tqdm
import itertools
from operator import add
import subprocess

COMMAND = """python -m byol.main_loop --experiment_mode='pretrain' --worker_mode='train' --pretrain_epochs=1000 """


def separator():
    print("-----------------------------------------")
    print("-----------------------------------------")
    print("-----------------------------------------")




headers = ["--batch_size=","--checkpoint_root="]

batch_sizes = [32,64,128,256,512,1024,2048]

#batch_sizes = [32]
all = [batch_sizes]

for el in tqdm(list(itertools.product(*all))):
    # Create model dir name and create dir for the model
    model_dir = "byol/models/"+str(el[0])+"-RN50"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # Join model dir name to the list of parameters
    list_el = [str(i) for i in el]
    model_dir_stringed = "\'"+model_dir+"\'"
    list_el.append(model_dir_stringed)
    # Create the list of parameters
    to_add = list(map(add,headers,list_el))
    # Create a string with the command that will be executed
    # In this iteration
    COMMAND_ITERATION = COMMAND + ' '.join(to_add)
    separator()
    print(COMMAND_ITERATION)
    os.system(COMMAND_ITERATION)
    


    

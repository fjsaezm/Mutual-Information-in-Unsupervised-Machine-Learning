import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tqdm import tqdm
import itertools
from operator import add
import subprocess

COMMAND = """python pretrain_eval.py --num_epochs 100 """


def separator():
    print("-----------------------------------------")
    print("-----------------------------------------")
    print("-----------------------------------------")




headers = ["--batch_size ",
            "--encoder ",
            "--logdir "]

batch_sizes = [1024,2048]
resnet_depths = ["resnet34"]

all = [batch_sizes,resnet_depths]

for el in tqdm(list(itertools.product(*all))):
    # Create model dir name and create dir for the model
    model_dir =  "models/" + '-'.join([str(elem) for elem in el])
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # Join model dir name to the list of parameters
    list_el = [str(i) for i in el]
    list_el.append(model_dir)
    # Create the list of parameters
    to_add = list(map(add,headers,list_el))
    # Create a string with the command that will be executed
    # In this iteration
    COMMAND_ITERATION = COMMAND + ' '.join(to_add)
    separator()
    print(COMMAND_ITERATION)
    os.system(COMMAND_ITERATION)
    


    

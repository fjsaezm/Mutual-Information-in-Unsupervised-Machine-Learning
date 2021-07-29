from tqdm import tqdm
import os
import itertools
from operator import add

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

for el in list(itertools.product(*all)):
    str_el = [str(i) for i in list(el)]
    print(str_el)
    model_dir =  '-'.join(map(str, str_el))
    print(model_dir)
    all_elements = str_el.append(model_dir)
    print(all_elements)
    print(headers)
    to_add = list(map(add,headers,all_elements))
    print(to_add)
    

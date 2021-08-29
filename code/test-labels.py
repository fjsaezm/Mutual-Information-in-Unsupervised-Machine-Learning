import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.image as image
from os import listdir
from os.path import isfile, join

from matplotlib.patches import Polygon



#Labels

# SimCLR Exp1
brown = mpatches.Patch(color='brown',label='16')
turquoise = mpatches.Patch(color='turquoise',label='32')
pink = mpatches.Patch(color='deeppink',label='64')
green = mpatches.Patch(color='teal',label='128')
gray = mpatches.Patch(color='lightgray',label='256')
orange = mpatches.Patch(color='orange',label='512')
d_blue = mpatches.Patch(color='dodgerblue',label='1024')

# SimCLR Exp2
royal_blue = mpatches.Patch(color='tab:blue', label='128')
pink = mpatches.Patch(color='deeppink',label='256')
brown = mpatches.Patch(color='tab:red',label='512')
green = mpatches.Patch(color='teal',label='1024')

# SimCLR Exp3
orange = mpatches.Patch(color='darkorange',label='Exp1')
royal_blue = mpatches.Patch(color='tab:blue', label='Exp2')
turquoise = mpatches.Patch(color='darkturquoise',label='Exp3-ResNet18')
tab_red = mpatches.Patch(color='tab:red',label='Exp3-ResNet50')

# BYOL Exp1

orange = mpatches.Patch(color='darkorange',label='32')
royal_blue = mpatches.Patch(color='tab:blue', label='64')
tab_red = mpatches.Patch(color='tab:red',label='128')
turquoise = mpatches.Patch(color='darkturquoise',label='256')
pink = mpatches.Patch(color='deeppink',label='512')
green = mpatches.Patch(color='teal',label='1024')

# BYOL Exp2
orange = mpatches.Patch(color='darkorange',label='RN18-32')
royal_blue = mpatches.Patch(color='tab:blue', label='RN18-64')
tab_red = mpatches.Patch(color='tab:red',label='RN50-128')
turquoise = mpatches.Patch(color='darkturquoise',label='RN50-256')


hands = [ orange,royal_blue,tab_red,turquoise]


onlyfiles = [f for f in listdir('.') if isfile(join('', f))]
images = [im for im in onlyfiles if im.endswith('.jpg')]
print(images)

for im in images:
    #im = 'train_total_loss.jpg'
    img = plt.imread(im)

    # drawing a random figure on top of your JPG
    fig,ax = plt.subplots()
    ax.imshow(img,extent=[0, 400, 0, 300])
    plt.legend(handles=hands,bbox_to_anchor=(1,0.8))
    name = im.split(".")[0]
    plt.axis('off')
    plt.savefig(name+".pdf")
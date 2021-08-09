import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.image as image

from matplotlib.patches import Polygon


im = 'train_supervised_acc.jpg'
img = plt.imread(im)
#Labels
red = mpatches.Patch(color='brown',label='16')
blue = mpatches.Patch(color='turquoise',label='32')
pink = mpatches.Patch(color='deeppink',label='64')
green = mpatches.Patch(color='teal',label='128')
gray = mpatches.Patch(color='lightgray',label='256')
orange = mpatches.Patch(color='orange',label='512')
d_blue = mpatches.Patch(color='dodgerblue',label='1024')

# drawing a random figure on top of your JPG
fig,ax = plt.subplots()
ax.imshow(img,extent=[0, 400, 0, 300])
plt.legend(handles=[red,blue,pink,green,gray,orange,d_blue],bbox_to_anchor=(1,0.6))
plt.axis('off')
plt.savefig("train_supervised_acc.pdf")
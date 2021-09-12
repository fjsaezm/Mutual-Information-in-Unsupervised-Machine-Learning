from os import listdir
from os.path import isdir
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


batches_18 = [32,64,128,256,512,1024]
batches_50 = [32,64,128,256]
accs_ols = [0.875,0.968,0.929,0.929,0.916,0.852]
accs_50 = [0.9225477,0.9240764,0.92203814,0.9202547]

df = pd.DataFrame({'batch-size':batches_50, 
    'label_top_1_accuracy':accs_50
})


ax = sns.pointplot(x="batch-size",y = "label_top_1_accuracy",data = df,color = "green")



# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.savefig("../thesis/media/byol-batch-comparison.pdf")
plt.show()
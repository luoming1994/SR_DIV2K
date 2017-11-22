import os
import numpy as np
from PIL import Image
import torch
import time
from DIV2K_DataSet import DIV2K_DataSet

import matplotlib.pyplot as plt

# argparse
class arg(object):
    def __init__(self,
              scale_list = [2],
              batch_size = 64,
              imagesDir = r'E:\Data\DIV2K',
              crop_size = 64):
        self.scale_list = scale_list
        self.batchSize = batch_size
        self.imagesDir = imagesDir
        self.crop_size = crop_size
        

opt = arg()
dataset = DIV2K_DataSet(data_dir = opt.imagesDir,
                            scale_list= opt.scale_list,
                        crop_size = opt.crop_size,
                        image_num = 2  )

data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = False)

img_hr_list = []
img_lr_list = []
for index,(images,labels) in enumerate(data_loader):
    for i in range(images.size()[0]):    # the lass batch image num may be less than  batchSize     
        img_lr = images[i,:,:].numpy()
        # convert numpy.array to Image.Image,image mode L         
        # img_lr = (255*img_lr).astype(np.uint8)
        # img_lr = Image.fromarray(img_lr)
        img_lr_list.append(img_lr)

        img_hr = labels[i,:,:].numpy()
        #img_hr = (255*img_hr).astype(np.uint8)
        #img_hr = Image.fromarray(img_hr)
        img_hr_list.append(img_hr)
		

#show
fig,axes = plt.subplots(4,4)
for i in range(2):
    for j in range(4):
        axes[2*i,j].imshow(img_hr_list[i*4+j],cmap=plt.cm.gray)
        axes[2*i+1,j].imshow(img_lr_list[i*4+j],cmap=plt.cm.gray)
#plt.axis('off')
plt.show()

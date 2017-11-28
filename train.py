# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from model.model import GenerateNet
from model.DIV2K import DIV2K_DataSet


import matplotlib.pyplot as plt

# argparse
class arg(object):
    def __init__(self,
                imagesDir   = r'E:\Data\DIV2K',
                imagesPth   = 'data.pth',   # data set pakage name
                netDir      = r'net',
                logDir      = r'log',
                scale_list  = [2],
                batch_size  = 64,
                crop_size   = 64,
                cuda        = True,
                niter       = 20):        
        self.imagesDir  = imagesDir
        self.imagesPth  = imagesPth     
        self.netDir     = netDir
        self.logDir     = logDir
        self.scale_list = scale_list
        self.batchSize  = batch_size
        self.crop_size  = crop_size
        self.cuda       = True
        self.niter      = niter
   
def calAcc(pred,label):
    """
    label:1 dim label,pred 2 dim torch.Tensor
    """
    # calculate acc
    pred_np = pred.cpu().data.numpy().argmax(axis=1)
    labels_np = label.cpu().data.numpy()
    acc = np.mean(pred_np ==labels_np)
    return acc

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)   
        
        
#  args
opt = arg(cuda=False)
#cudnn.benchmark = True
cudnn.enabled = False

if os.path.exists(os.path.join(opt.imagesDir,opt.imagesPth)):
    dataset = torch.load(os.path.join(opt.imagesDir,'data.pth'))
else:
    dataset = DIV2K_DataSet(data_dir = opt.imagesDir,
                            scale_list= opt.scale_list,
                            crop_size = opt.crop_size,
                            image_num = 10  )
    # save the imagedata
    torch.save(dataset,os.path.join(opt.imagesDir,'data.pth'))
data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = True)

                                           
netG = GenerateNet()
#netG.apply(weights_init)
if(opt.cuda):
    netG.cuda()

optimizer = optim.Adam(netG.parameters(),lr=0.001,weight_decay =0.1)
#optimizer = optim.SGD(netG.parameters(),lr=0.001,momentum=0.9) 
criterion = nn.MSELoss()
netG.train()
t0 = time.time()
with open(os.path.join(opt.logDir,'train.log'),'a') as f:
    loss_list= []
    for epoch in range(opt.niter):#epoch
        for indx,(images,labels) in enumerate(data_loader):#batch
            netG.zero_grad()
            sub = (images - labels).numpy()
            if(opt.cuda):
                images = images.cuda()
                labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)

            out = netG(images)
            err = criterion(out,labels)
            err.backward()
            optimizer.step()
    
            # write loss into file 
            write_str = '%f\n'%(err.data[0])
            f.write(write_str)
            loss_list.append(err.data[0])
            #print mean loss in 100 batch
            if len(loss_list) == 1:
                print('[train][%d/%d]Loss: %.4f'%(epoch, opt.niter,sum(loss_list)/len(loss_list)))
                loss_list[:] = []
print("train caost:%.4f"%(time.time()-t0))
torch.save(netG.state_dict(), r'%s/netG.pth' % (opt.netDir))                                        

if False:                                          
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
#fig,axes = plt.subplots(4,4)
#for i in range(2):
#    for j in range(4):
#        axes[2*i,j].imshow(img_hr_list[i*4+j],cmap=plt.cm.gray)
#        axes[2*i+1,j].imshow(img_lr_list[i*4+j],cmap=plt.cm.gray)
#plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:22:17 2017

@author: LM
"""

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
from model.DIV2K import DIV2K_patchDataSet


# argparse
class arg(object):
    def __init__(self,
                imgDir = r'E:\Data\DIV2K',
                netSaveDir  = r'net',
                logOutDir   = r'log',
                scale_list = [2],
                batch_size = 4,
                cuda = True,
                niter = 20):
        
        self.imgDir     = imgDir
        self.netSaveDir = netSaveDir
        self.logOutDir  = logOutDir
        self.scale_list = scale_list
        self.batchSize  = batch_size
        self.cuda       = True
        self.niter      = niter
        
opt = arg(cuda = False)
train_set = DIV2K_patchDataSet(opt.imgDir,
                             scale_list= opt.scale_list,
                             img_num = 1024  )
train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                           batch_size = opt.batchSize,
                                           shuffle = True)
cudnn.enabled = False
netG = GenerateNet()
if(opt.cuda):
    netG.cuda()

optimizer = optim.Adam(netG.parameters(),lr=0.001,weight_decay =0.1)
#optimizer = optim.SGD(netG.parameters(),lr=0.001,momentum=0.9) 
criterion = nn.MSELoss()
netG.train()
t0 = time.time()
with open(os.path.join(opt.logOutDir,'trainPatch.log'),'a') as f:
    loss_list= []
    for epoch in range(opt.niter):#epoch
        for indx,(images,labels) in enumerate(train_loader):#batch
            netG.zero_grad()
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
                print('[train][epoch:%d/%d][batch:%d]Loss: %.4f'%(epoch, opt.niter,indx,sum(loss_list)/len(loss_list)))
                loss_list[:] = []
print("train caost:%.4f"%(time.time()-t0))
torch.save(netG.state_dict(), '%s/netG.pth' % (opt.netSaveDir))    
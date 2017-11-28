# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:17:17 2017

@author: LM
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image

from model.model import GenerateNet


def loadImgAsYCbCr(imgPath,upscale=None):
    """
    load a image and convert to YCbCr,return Y,Cb,Cr channels
    upscale:upscale the image with bicubic function
    """
    img = Image.open(imgPath)
    if upscale is not None:
        W,H = img.size
        img = img.resize((W*upscale, H*upscale),Image.BICUBIC)
    img_YCbCr = img.convert('YCbCr')
    Y, Cb, Cr = img_YCbCr.split()
    return Y,Cb,Cr

def mergeYCbCrImg2RGB(Y,Cb,Cr):
    """
    merge Y,Cb,Cr channels into a image and convert RGB mode
    """
    img_YCbCr = Image.merge('YCbCr',(Y,Cb,Cr))
    img_rgb = img_YCbCr.convert('RGB')
    return img_rgb

def PSNR(im,gt):
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1
    mse = np.mean((gt - im)**2)
    psnr = 10*np.log10(255**2/mse)
    return psnr

def SSIM(im,gt):
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1   
    
    # C1=(K1*L)^2, 
    # C2=(K2*L)^2
    # C3=C2/2,     1=0.01, K2=0.03, L=255
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    C3 = C2/2.0
    
    mean_x = im.mean() # mean of im
    mean_y = gt.mean() # mean of gt
    cov = np.cov([gt.flatten(),im.flatten()])
    cov_xx = cov[0,0]
    cov_x = np.sqrt(cov_xx)
    cov_yy= cov[1,1]
    cov_y = np.sqrt(cov_yy) 
    cov_xy = cov[0,1]
    
    l_xy = (2*mean_x*mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    c_xy = (2*cov_x*cov_y + C2) / (cov_xx + cov_yy + C2)
    s_xy = (cov_xy + C3) / (cov_x*cov_y + C3)
    ssim = l_xy*c_xy*s_xy
    
    return ssim
    
    
# argparse
class arg(object):
    def __init__(self,
                net_path        = r'net/GenerateNet.pth',
                image_path_lr   = '',
                image_path_hr   = '',
                upscale         = 2,
                saveImage       = True,
                save_path       = r'E:\Data\DIV2K\5SR.bmp',
                cuda            = True):
        
        self.net_path       = net_path    
        self.image_path_lr = image_path_lr
        self.image_path_hr = image_path_hr
        self.upscale        = upscale
        self.saveImage      = saveImage
        self.save_path      = save_path
        self.cuda           = True

#set argparse        
opt = arg(cuda = False,
        image_path_lr = r'E:\Data\DIV2K\5LR.bmp',
        image_path_hr = r'E:\Data\DIV2K\5HR.bmp'
        )
#load the network
netG = GenerateNet()
netG.load_state_dict(torch.load(opt.net_path))
#load image
#Y_lr,Cb_lr,Cr_lr = loadImgAsYCbCr(opt.image_path_lr,upscale = opt.upscale)
Y_lr,Cb_lr,Cr_lr = loadImgAsYCbCr(opt.image_path_lr)
Y_hr,Cb_hr,Cr_hr = loadImgAsYCbCr(opt.image_path_hr)
# convert image => numpy.array => torch.Tensor => Varible
y_in = np.array(Y_lr,dtype=np.float32)/255.
img_in = torch.Tensor(y_in)
img_in = torch.unsqueeze(img_in,0)# 2dim => 3dim
img_in = torch.unsqueeze(img_in,0)# 3dim => 4dim
img_in = Variable(img_in)
# forward the net ,get output
Y_out = netG(img_in)
# convert output data type
Y_out_np = np.array(Y_out.data[0,0].numpy()*255,dtype = np.float32) # 4 dim Tensor => 2dim numpy.array
Y_out_np[Y_out_np<0]=0          # [0,255]
Y_out_np[Y_out_np>255]=255
Y_sr  = Image.fromarray(np.array(Y_out_np,dtype = np.uint8))   # 2dim array =>  L mode Image
img_sr = mergeYCbCrImg2RGB(Y_sr,Cb_lr,Cr_lr)
img_sr.save(opt.save_path)
#vutils.save_image(Y_sr.data,opt.save_path,normalize=True)
lr_psnr = PSNR(np.array(Y_hr,dtype = np.float32),np.array(Y_lr,dtype = np.float32))
sr_psnr = PSNR(np.array(Y_hr,dtype = np.float32),Y_out_np)
PSNR(np.array(Y_hr,dtype = np.float32),np.array(Y_out.data[0,0].numpy()*255,dtype = np.float32))
print(lr_psnr,sr_psnr)



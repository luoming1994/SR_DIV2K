# -*- coding: utf-8 -*-
"""
Created on Thu Des 1 15:17:17 2017

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

def is_image_file(filename):
	"""
	a file is a image? via extension
	"""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

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

def getLoss(net,img_lr_path,img_hr_path):
    """
        get loss 
    """
    
def validModel(net,img_lr,img_hr,save_dir = None):
    """
    get loss of valid data 
    net: trained model
    img_lr: lr image file path or directory
    img_hr: hr image file path or directory
    save_dir: directory to save the super reselution image of the lr image through the net
    """
    # get the lr and hr images paths list
    if  not os.path.exists(img_lr):
        raise Exception("%s is not exist !"%img_lr)
    if  not os.path.exists(img_hr):
        raise Exception("%s is not exist !"%img_hr)
    img_lr_paths = []
    img_hr_paths = []
    if os.path.isfile(img_lr) and os.path.isfile(img_hr):
        img_lr_paths.append(img_lr)
        img_hr_paths.append(img_hr)
    elif os.path.isdir(img_lr) and os.path.isdir(img_hr):
        img_lr_paths = [os.path.join(img_lr, x) for x in os.listdir(img_lr) if is_image_file(x)]
        img_hr_paths = [os.path.join(img_hr, x) for x in os.listdir(img_hr) if is_image_file(x)]
    assert  len(img_lr_paths) != len(!img_hr_paths), \ 
    "number of the %s and %s is not the same!"%(img_lr_paths,img_hr_paths)
    lr_loss_list = []
    sr_loss_list = []
    Y_lr,
    for index in range(len(img_lr_paths)):
        Y_lr,Cb_lr,Cr_lr = loadImgAsYCbCr(img_lr_paths[index])
        Y_hr,Cb_hr,Cr_hr = loadImgAsYCbCr(img_lr_paths[index])
        # convert image => numpy.array => torch.Tensor => Varible
        y_lr_np = np.array(Y_lr,dtype = np.float32)
        Y_hr_np = np.array(Y_hr,dtype = np.float32)
        net_in = y_lr_np/255.       # 0~255 => 0~1
        net_in = net_in.reshape(1,1,net_in.shape[-2],net_in.shape[-1])  # 2dim =>4dim
        net_in = torch.Tensor(net_in)   # numpy.array => torch.Tensor
        #img_in = torch.unsqueeze(img_in,0)# 2dim => 3dim
        net_in = Variable(net_in)
        # forward the net ,get output
        net_out = net(net_in)
        # 4 dim Tensor => 2dim numpy.array
        net_out_np = np.array(net_out.data[0,0].numpy()*255,dtype = np.float32) 
        net_out_np[net_out_np<0]=0          # [0,255]
        net_out_np[net_out_np>255]=255
        if save_dir:
            Y_sr  = Image.fromarray(np.array(net_out_np,dtype = np.uint8))   # 2dim array =>  L mode Image
            img_sr = mergeYCbCrImg2RGB(Y_sr,Cb_lr,Cr_lr)
            path,name = os.path.split(img_lr_paths[index])
            img_sr.save(os.path.join(save_dir,name))
        #vutils.save_image(Y_sr.data,opt.save_path,normalize=True)
        
        lr_psnr = PSNR(Y_hr_np,y_lr_np)
        sr_psnr = PSNR(Y_hr_np,net_out_np)
        lr_loss_list.append(lr_psnr)
        sr_loss_list.append(sr_psnr)
    # get mean loss of all the valid data
    lr_psnr_mean = np.mean(lr_loss_list)
    lr_loss_list.append(lr_psnr_mean)
    lr_psnr_mean = np.mean(sr_loss_list)
    sr_loss_list.append(lr_psnr_mean)
    return lr_loss_list,sr_loss_list
        
        
 
    
    
    
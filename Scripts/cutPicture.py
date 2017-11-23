# -*- coding: utf-8 -*- #

import os
from PIL import Image


def cutImage(laodDir = r'',
			saveDir = r'',
			upscale = 1,
			num_start = 1,
			num_end = 800,
			cut_size = 64):
	"""
	Image_DIR_HR: ~
	Image_DIR_LR: ~/X2
					~/X3
	cut image into cut_size*cut_size patchs
	"""
	index = 0
	for i in range(num_start,num_end+1,1):
		if upscale >= 2: 
			path = os.path.join(laodDir,'X%d'%upscale,'%04dx%d.png'%(i,upscale))
		else:
			path = os.path.join(laodDir,'%04d.png'%i)			
		img = Image.open(path)
		if upscale >= 2:
			img = img.resize((img.size[0]*upscale,img.size[1]*upscale),Image.BICUBIC)    
		Width,Height = img.size
		W_num,H_num = Width//cut_size,Height//cut_size
		for h in range(H_num):
			for w in range(W_num):
				box = (w*cut_size,h*cut_size,(w+1)*cut_size,(h+1)*cut_size)
				img_cut = img.crop(box)
				if upscale >= 2: 
					save_path = os.path.join(saveDir,'X%d'%upscale,'%08d.png'%index)
				else:
					save_path = os.path.join(saveDir,'%08d.png'%index)		
				img_cut.save(save_path,'png')
				index = index + 1
				if index % 10000 ==0:
					print('%d images done!'%index)
	print('All the %d images done!'%index)

if __name__ == '__main__':	
	laodDir_HR = r'E:\Data\DIV2K\DIV2K_train_HR'
	saveDir_HR = r'E:\Data\DIV2K\DIV2K_patch_train_HR'
	cutImage(laodDir_HR,saveDir_HR,num_start=1,num_end=2)

	loadDir_LR = r'E:\Data\DIV2K\DIV2K_train_LR_bicubic'
	saveDir_LR = r'E:\Data\DIV2K\DIV2K_patch_train_LR'
	for scale in [2,3]:
	    cutImage(loadDir_LR,saveDir_LR ,upscale = scale,num_start=1,num_end=2)
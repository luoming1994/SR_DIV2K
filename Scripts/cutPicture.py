import os
from PIL import Image


def cutImage(laodDir = r'',
			saveDir = r'',
			upscale = 1,
			num_start = 1,
			num_end = 800,
			cut_size = 64):
	index = 0
	for i in range(num_start,num_end+1,1):
		path = os.path.join(laodDir,'%04d.png'%i) 			
		img = Image.open(path)
		Width,Height = img.size
		W_num,H_num = Width//cut_size,Height//cut_size
		for h in range(H_num):
			for w in range(W_num):
				box = (w*cut_size,h*cut_size,(w+1)*cut_size,(h+1)*cut_size)
				img_cut = img.crop(box)
				save_path = os.path.join(saveDir,'%08d.png'%index)
				img_cut.save(save_path,'png')
				index = index + 1
	
	
laodDir = r'E:\Data\DIV2K\DIV2K_train_HR'
saveDir = r'E:\Data\DIV2K\DIV2K_patch_train_HR'
cutImage(laodDir,saveDir = saveDir,num_start=1,num_end=2)

loadDir_LR = r'E:\Data\DIV2K\DIV2K_train_LR_bicubic\X2'
saveDir_LR = r'E:\Data\DIV2K\DIV2K_patch_train_LR\X2'
cutImage(laodDir,saveDir = saveDir,num_start=1,num_end=2)
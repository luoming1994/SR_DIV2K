from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

 
def is_image_file(filename):
	"""
	a file is a image? via extension
	"""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath,upscale=1):
	"""
	load a image and convert to YCbCr,return Y channel
	"""
    img = Image.open(filepath)
	img_crop = crop_img(img,upscale)
	img_YCbCr = img_crop.convert('YCbCr')
    y, _, _ = img_YCbCr.split()
    return y

def crop_img(img,scale_num):
	"""
	crop image with integer multiple of scale_num
	"""
	if isinstance(img, Image.Image):
		W,H = img.size
		W_crop , H_crop = W//scale_num , H//scale_num
		img_crop = img.crop((0,0,W_crop,H_crop))  

	return img_crop
	

def cut_img(img,crop_size = 64):
	"""
	cut image(torch.Tensor) into normal size with size(crop_size * crop_size)
	"""
	if isinstance(img, torch.Tensor):
		img_size = img.size()
		assert len(img_size)==3 
		if img_size[-1]>=64 and img_size[-2]>=64:
			H,W = img_size[-2]//crop_size,img_size[-2]//crop_size
			chnnl = img_size[0]   # img channel
			crop_img  = torch.Tensor(H*W*chnnl,crop_size,crop_size)
			for h in range(0,H,1):
				for w in range(0,W,1):
					indx = h*W+w
					crop_img[indx*chnnl:(indx+1)*chnnl,:,:] = img[:,h*64:(h+1)*64,w*64:(w+1)*64]
	
    return crop_img

class MyDataSet(data.Dataset):
	"""
	super resolution image dataset
	image_dir: images file dir
	input_transform:
	target_transform:
	"""
	def __init__(self, image_dir,upscale):
		super(MyDataSet, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        #self.input_transform = input_transform
        #self.target_transform = target_transform
		# load all image
		self.data = torch.Tensor()
		for filename in image_filenames:
			image = load_img(filename)
			image = transforms.ToTensor()(image)
			crop_img = cut_img(image)
			self.data = torch.cat((self.data,crop_img),dim=0)
			
		

    def __getitem__(self, index):
        
		#input = load_img(self.image_filenames[index])
        #target = input.copy()
        #if self.input_transform:
		#    # 1*H*W
        #    input = self.input_transform(input)
        #if self.target_transform:
        #    target = self.target_transform(target)
		# H*W 2 dim
		input = self.data[indx]
		target= self.label[indx]
		
        return input, target

    def __len__(self):
        #return len(self.image_filenames)
		return self.data.size()[0]
		
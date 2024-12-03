'''
Denne kode er taget fra https://github.com/ykdai/Flare7K/tree/main/Generate_flare_on_light 
Men omskrevet for at implementeres i CLARITY_dataloader.py

'''

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import random
from scipy import ndimage
from skimage import morphology
from skimage.measure import label
from skimage.filters import rank
from skimage.morphology import disk
from skimage import color
from skimage.measure import regionprops
import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch

random.seed(42)

def plot_light_pos(input_img,threshold):
	#input should be a three channel tensor with shape [C,H,W]
	#Out put the position (x,y) in int
	luminance=0.3*input_img[0]+0.59*input_img[1]+0.11*input_img[2] # her beregner den luminance af billedet baseret på luminance equation som har vægte for hvor meget mennesker ser hver farve
	luminance_mask=luminance>threshold # Den her sat et threshold og for pixel der overskrider der laver den en luminance mask
	luminance_mask_np=luminance_mask.numpy() # Den gør masken til numpy
	struc = disk(3) #
	img_e = ndimage.binary_erosion(luminance_mask_np, structure = struc)
	img_ed = ndimage.binary_dilation(img_e, structure = struc)

	labels = label(img_ed)
	if labels.max() == 0:
		# print("Light source not found.")
		x = random.randint(0, 255)
		y = random.randint(0, 255)

		return (x, y)
	else:
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
		largestCC=largestCC.astype(int)
		properties = regionprops(largestCC, largestCC)
		weighted_center_of_mass = properties[0].weighted_centroid
		print("Light source detected in position: x:",int(weighted_center_of_mass[1]),",y:",int(weighted_center_of_mass[0]))
		light_pos = (int(weighted_center_of_mass[1]),int(weighted_center_of_mass[0]))
		light_pos=[light_pos[0]-256,light_pos[1]-256]

		return light_pos

class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma #Den instantierer lokal variabel for gamma
	def __call__(self,image): # Når en instans af klassen bliver kaldt kaldes den her funktion
		if self.gamma == None: # Den sætter sin egen gamma hvis du ikke har sat en
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else: # Vi har sat gamma til så derfor vil denne blive kaldt
			return TF.adjust_gamma(image,self.gamma,gain=1) # den føjer gamma baseret på den satta gamma værdi

class TranslationTransform(object):
    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x,angle=0, scale=1,shear=[0,0], translate= list(self.position))

def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image

class Flare_Image_Loader(data.Dataset):
	def __init__(self, base_image ,transform_base=None,transform_flare=None,mask_type=None):
		self.base_image = base_image #base image er billedet som flare skal tilføjes på

		self.ext = ['png','jpeg','jpg','bmp','tif'] #Den vil kun tilføje en flare til et billede med de her fil typer
		self.flare_dict={} # Den føjer et dictionary der holder flares
		self.flare_list=[] # Den føjer en liste der holder flares
		self.flare_name_list=[] # Den føjer en list der holder navnene til flare 
								# Så flare list og flare name list er to lister der opgør det samme som flare_dict
		self.reflective_flag=False #Den føjer en indelende bool om hvis der er føjet en reflective flare eller ej
		self.reflective_dict={} #Lignende logik for reflective flares som til flares
		self.reflective_list=[]
		self.reflective_name_list=[]

		self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"
									# ([HVAD GØR DEN])

		self.transform_base=transform_base #Her føjes en variabel til hvis en bruger af koden har deres egen transform_base
		self.transform_flare=transform_flare #Ligeledes for specifikke transformer til flare

		# print("Base Image Loaded with examples:", len(self.data_list))

	def apply_flare(self):
		gamma=np.random.uniform(1.8,2.2) # Den vælger en random værdi mellem de to tal og der er lige sandsynlighed for alle tal imellem dem
		to_tensor=transforms.ToTensor() #ToTensor funktionen bliver føjet til variabel
		adjust_gamma=RandomGammaCorrection(gamma) # Den bruger gamme hvilket var et tal mellem de to tal i uniform og bruger i gammacorrection

		adjust_gamma_reverse=RandomGammaCorrection(1/gamma) #Definerer gamma reverse halløj. 1/gamma bcuz math
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)  #Den her tilføje random jitter
		if self.transform_base is not None: # Hvis der er en base transform der er tilføjer til Flare_Image_loader bliver den brug
			self.base_image=to_tensor(self.base_image)
			self.base_image=adjust_gamma(self.base_image)
			self.base_image=self.transform_base(self.base_image)
		else: # Ellers bliver base bare .permute(2,0,1) hvilket shuffler dimensionerne af tensoren
			self.base_image=to_tensor(self.base_image)
			self.base_image=adjust_gamma(self.base_image)
			self.base_image=self.base_image.permute(2,0,1)
		sigma_chi=0.01*np.random.chisquare(df=1) #Den giver mig et tal fra chi squared distribution
		self.base_image=Normal(self.base_image,sigma_chi).sample() #Tilføjer basically bare støj men på en super klog math måde
		gain=np.random.uniform(1,1.2) # Finder et random tal mellem 1 og 1,2
		flare_DC_offset=np.random.uniform(-0.02,0.02) # Finder et random tal mellem -0.02 og 0.02
		self.base_image=gain*self.base_image #Den ganger alle pixel værdier i billedet med gain for at gøre det mere bright
		self.base_image=torch.clamp(self.base_image,min=0,max=1) #sørger for at alle pixel værdier er mellem 0 og 1

		light_pos=plot_light_pos(self.base_image,0.97**gamma) #Den returnerer en position og hvis der er en lyskilde på billedet vil den finde den. ellers vil den returnere en random position
				#traslate=TranslationTransform(light_pos)
		transform_flare=transforms.Compose([transforms.RandomHorizontalFlip(), #Det her vender billedet højre til venstre med 50% chance
							  transforms.RandomVerticalFlip(), # Vender det op eller ned med samme sandsynlighed
                              transforms.RandomAffine(degrees=(0,360),scale=(0.4,0.8),translate=(0,0),shear=(-20,20)),  # roteret det med en random grad mellem 0 og 360. skalerer flare med et random tal mellem 0.8 og 1.5. og applier et shear
							  TranslationTransform(light_pos)
                            #  ([DEN HER ER FJERNET FORDI DET SKABTE EN WIERD GRÅ FIRKANT AT CROPPE DET 2 GANGE]) --> transforms.CenterCrop((512,512)),
                              ])

		#load flare image
		flare_path=random.choice(self.flare_list) #Den vælger en random flare
		flare_img =Image.open(flare_path)
		if self.reflective_flag: # Det her er true hvis der er loadet en reflective flare 
			reflective_path=random.choice(self.reflective_list)
			reflective_img =Image.open(reflective_path)

		flare_img=to_tensor(flare_img)
		flare_img=adjust_gamma(flare_img)
		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img) #Den adjuster ligeledes gamma på reflective
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1) # Den ligge scatter flare og reflective flare i et billede

		flare_img=remove_background(flare_img) # Den forstørrer basically kontrasten så der er større forskel mellem flare og baggrunden

		if self.transform_flare is not None:
			flare_img=self.transform_flare(flare_img)
		else: # Den bruger her den transorm der vender og drejer flare billedet
			flare_img=transform_flare(flare_img)
		
		#change color
		flare_img=color_jitter(flare_img) # Så tilføjet den random farve spikes

		#Det her tilføjer blur på flare billedet. 
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0)) 
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset # Det her tilføjer eller fjerner en lille smule intensitet (pixelværdi)
		flare_img=torch.clamp(flare_img,min=0,max=1)

		#merge image	
		print(flare_img.shape)
		print(self.base_image.shape)

		if self.base_image.dim() == 3 and self.base_image.shape[0] != 3: #Det her sørger for at channel dim er det rigtige sted
			self.base_image = self.base_image.permute(1, 0, 2)

		base_height, base_width = self.base_image.shape[1], self.base_image.shape[2]

		# Update flare transformations to match base image size
		transform_flare = transforms.Compose([ # Der bliver processeret flares igen men nu bliver de croppet så det passer base image
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomAffine(
				degrees=(0, 360),
				scale=(0.8, 1.5),
				translate=(0, 0),
				shear=(-20, 20),
			),
			TranslationTransform(light_pos),
			transforms.CenterCrop((base_height, base_width)),
		])

		# Apply transformations to flare image
		flare_img = transform_flare(flare_img)

		# Ensure flare_img and self.base_image have the same size
		if flare_img.shape[1:] != self.base_image.shape[1:]:
			# Resize flare_img to match base_image
			resize_transform = transforms.Resize((base_height, base_width))
			flare_img = resize_transform(flare_img)

		merge_img=flare_img+self.base_image # de samler de to billeder ved at plusse værdierne for de to tensors
		merge_img=torch.clamp(merge_img,min=0,max=1)

		flare =Image.open(flare_path)

		if self.mask_type==None:
			return adjust_gamma_reverse(self.base_image),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma, flare
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(self.base_image)
			zero = torch.zeros_like(self.base_image)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(self.base_image)
			zero = torch.zeros_like(self.base_image)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)
		return adjust_gamma_reverse(self.base_image),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),flare_mask,gamma, flare
	







	def apply_flare_with_flare(self, flare):
		gamma=np.random.uniform(1.8,2.2)
		to_tensor=transforms.ToTensor()
		adjust_gamma=RandomGammaCorrection(gamma)
		adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
		if self.transform_base is not None:
			self.base_image=to_tensor(self.base_image)
			self.base_image=adjust_gamma(self.base_image)
			self.base_image=self.transform_base(self.base_image)
		else:
			self.base_image=to_tensor(self.base_image)
			self.base_image=adjust_gamma(self.base_image)
			self.base_image=self.base_image.permute(2,0,1)
		sigma_chi=0.01*np.random.chisquare(df=1)
		self.base_image=Normal(self.base_image,sigma_chi).sample()
		gain=np.random.uniform(1,1.2)
		flare_DC_offset=np.random.uniform(-0.02,0.02)
		self.base_image=gain*self.base_image
		self.base_image=torch.clamp(self.base_image,min=0,max=1)

		light_pos=plot_light_pos(self.base_image,0.97**gamma)
		
		light_pos=[light_pos[0]-256,light_pos[1]-256]
		#traslate=TranslationTransform(light_pos)
		transform_flare=transforms.Compose([transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip(),
                              transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(0,0),shear=(-20,20)),
							  TranslationTransform(light_pos),
                              transforms.CenterCrop((512,512)),
                              ])

		#load flare image
		flare_img = flare
		if self.reflective_flag:
			reflective_path=random.choice(self.reflective_list)
			reflective_img =Image.open(reflective_path)

		flare_img=to_tensor(flare_img)
		flare_img=adjust_gamma(flare_img)
		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img)
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

		flare_img=remove_background(flare_img)

		if self.transform_flare is not None:
			flare_img=self.transform_flare(flare_img)
		else:
			flare_img=transform_flare(flare_img)
		
		#change color
		flare_img=color_jitter(flare_img)

		#flare blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset
		flare_img=torch.clamp(flare_img,min=0,max=1)

		#merge image	
		if self.base_image.dim() == 3 and self.base_image.shape[0] != 3:
			self.base_image = self.base_image.permute(1, 0, 2)
		base_height, base_width = self.base_image.shape[1], self.base_image.shape[2]

		# Update flare transformations to match base image size
		transform_flare = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomAffine(
				degrees=(0, 360),
				scale=(0.8, 1.5),
				translate=(0, 0),
				shear=(-20, 20),
			),
			TranslationTransform(light_pos),
			transforms.CenterCrop((base_height, base_width)),
		])

		# Apply transformations to flare image
		flare_img = transform_flare(flare_img)

		# Ensure flare_img and self.base_image have the same size
		if flare_img.shape[1:] != self.base_image.shape[1:]:
			# Resize flare_img to match base_image
			resize_transform = transforms.Resize((base_height, base_width))
			flare_img = resize_transform(flare_img)

		merge_img=flare_img+self.base_image
		merge_img=torch.clamp(merge_img,min=0,max=1)

		if self.mask_type==None:
			return adjust_gamma_reverse(self.base_image),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(self.base_image)
			zero = torch.zeros_like(self.base_image)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(self.base_image)
			zero = torch.zeros_like(self.base_image)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)
		return adjust_gamma_reverse(self.base_image),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),flare_mask,gamma
			
	def load_scattering_flare(self,flare_name,flare_path):
		flare_list=[]
		[flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
		self.flare_name_list.append(flare_name)
		self.flare_dict[flare_name]=flare_list
		self.flare_list.extend(flare_list)
		len_flare_list=len(self.flare_dict[flare_name])
		if len_flare_list == 0:
			print("ERROR: scattering flare images are not loaded properly")
		else:
			print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
		print("Now we have",len(self.flare_list),'scattering flare images')

	def load_reflective_flare(self,reflective_name,reflective_path):
		self.reflective_list=[]
		self.reflective_flag=True
		[self.reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]

		self.reflective_name_list.append(reflective_name)
		self.reflective_dict[reflective_name]=self.reflective_list
		self.reflective_list.extend(self.reflective_list)
		len_reflective_list=len(self.reflective_dict[reflective_name])
		if len_reflective_list == 0:
			print("ERROR: reflective flare images are not loaded properly")
		else:
			print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
		print("Now we have",len(self.reflective_list),'refelctive flare images')



if __name__ == "__main__":
	base_image = r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Data\LOLdataset\our485\low\6.png"
	input_flare_image_loader = Flare_Image_Loader()
	input_flare_image_loader.load_scattering_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Scattering_Flare\Light_Source")
	input_flare_image_loader.load_reflective_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Reflective_Flare")
	_,_,input_image,_, flare=input_flare_image_loader.apply_flare()



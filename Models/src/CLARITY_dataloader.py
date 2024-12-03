from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / 'data'
sys.path.append(str(data_path))

from APPLY_FLARE_TO_IMG import Flare_Image_Loader

class LolDatasetLoader():
    def __init__(self, flare=bool, light_source_on_target=bool):
        self.light_source_on_target = light_source_on_target
        self.flare = flare
        self.root = r"C:\Users\Victor Steinrud\Downloads\lol_dataset\our485"
        self.inputs_dir = os.path.join(self.root, r'low')
        self.targets_dir = os.path.join(self.root, r'high')
        self.transform = transforms.ToTensor()

        self.inputs = sorted(os.listdir(self.inputs_dir))
        self.targets = sorted(os.listdir(self.targets_dir))



    def __len__(self): # find docs: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html -- 'Creating a Custom Dataset for your files'
        return len(self.inputs)
    
    def __getitem__(self, idx):

        input_path = os.path.join(self.inputs_dir, self.inputs[idx])
        target_path = os.path.join(self.targets_dir, self.targets[idx])
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        if self.flare and self.light_source_on_target:
            input_flare_image_loader=Flare_Image_Loader(input_image,transform_base=None,transform_flare=None)
            input_flare_image_loader.load_scattering_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Scattering_Flare\Light_Source")
            input_flare_image_loader.load_reflective_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Reflective_Flare")
            _,_,input_image,_, flare=input_flare_image_loader.apply_flare()

            target_flare_image_loader=Flare_Image_Loader(target_image,transform_base=None,transform_flare=None)
            target_flare_image_loader.load_scattering_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Scattering_Flare\Light_Source")
            _,_,target_image,_=target_flare_image_loader.apply_flare_with_flare(flare)
            input_image = input_image.transpose(1, 2)
            target_image = target_image.transpose(1, 2)


        if self.flare and not self.light_source_on_target:
            flare_image_loader=Flare_Image_Loader(input_image,transform_base=None,transform_flare=None)
            flare_image_loader.load_scattering_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Scattering_Flare\Light_Source")
            flare_image_loader.load_reflective_flare('Flare7K',r"C:\Users\Victor Steinrud\Downloads\Reflective_Flare")
            _,_,input_image,_,flare=flare_image_loader.apply_flare()

            target_image = self.transform(target_image) 
            input_image = input_image.transpose(1, 2)

        if not self.flare and not self.light_source_on_target:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
    
if __name__ == "__main__":
    l = LolDatasetLoader(flare=True, light_source_on_target=False)
    train_loader = DataLoader(l)
    for input, target in train_loader:
        input = input.squeeze(0)
        target = target.squeeze(0)
        input_image = input.permute(1, 2, 0).detach().cpu().numpy()
        target_image = target.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(input_image)
        plt.show()
        plt.imshow(target_image)
        plt.show()
        
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / 'data'
sys.path.append(str(data_path))

from APPLY_FLARE_TO_IMG import Flare_Image_Loader

class LolDatasetLoader(Dataset):
    def __init__(self, flare: bool):
        self.flare = flare
        self.root = r"Data/" #Data/LOLdataset/our485"
        self.inputs_dirs = [r'Data/LOLdataset/our485/low', r"Data/LOL-v2/Synthetic/Train/Low", r"Data/LOL-v2/Real_captured/Train/Low"] #os.path.join(self.root, r'low')
        self.targets_dirs = [r'Data/LOLdataset/our485/high', r"Data/LOL-v2/Synthetic/Train/Normal", r"Data/LOL-v2/Real_captured/Train/Normal"] #os.path.join(self.root, r'high')
        self.transform = transforms.ToTensor()

        included_extenstions = ['png']
        
        self.inputs = []
        for input_dir in self.inputs_dirs:
            input_images = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if any(file.endswith(ext) for ext in included_extenstions)]
            self.inputs.extend(input_images)

        self.targets = []
        for target_dir in self.targets_dirs:
            target_images = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if any(file.endswith(ext) for ext in included_extenstions)]
            self.targets.extend(target_images)
        print(self.targets)



    def __len__(self): # find docs: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html -- 'Creating a Custom Dataset for your files'
        return len(self.inputs)
    
    def __getitem__(self, idx):

        scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"

        input_path = self.inputs[idx]
        target_path = self.targets[idx]
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        if self.flare: 
            flare_image_loader=Flare_Image_Loader(input_image,transform_base=None,transform_flare=None)
            flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)
            _,_,input_image,_,flare=flare_image_loader.apply_flare()

            target_image = self.transform(target_image) 
            input_image = input_image.transpose(1, 2)

        else:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
    
if __name__ == "__main__":
    l = LolDatasetLoader(flare=True)
    train_loader = DataLoader(l, batch_size=1)
    for input, target in train_loader:
        input = input.squeeze(0)
        target = target.squeeze(0)
        input_image = input.permute(1, 2, 0).detach().cpu().numpy()
        target_image = target.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(input_image)
        plt.show()
        plt.imshow(target_image)
        plt.show()
        
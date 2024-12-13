import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from Modules.Preprocessing.preprocessing import preprocessing_pipeline_example, crop_flip_pipeline, cropping_only_pipeline, random_crop_and_flip_pipeline, resize_pipeline
import torch
import torchvision.transforms.functional as F

data_path = Path(__file__).resolve().parent.parent / 'data'
sys.path.append(str(data_path))

from Models.src.APPLY_FLARE_TO_IMG import Flare_Image_Loader

# Takes input and output and ensures the same transformation is done to both of them
# Helps when some tranformations are random, ensuring same transformation is done to input and target
def identical_transform(transform, input, target):
    concat = torch.cat((input.unsqueeze(0), target.unsqueeze(0)),0)
    transformed = transform(concat)
    input_trans = transformed[0]
    target_trans = transformed[1]
    return input_trans, target_trans


def get_images(dir, included_extenstions):
    files = []
    for d in dir:
        input_images = [os.path.join(d, file) for file in os.listdir(d) if any(file.endswith(ext) for ext in included_extenstions)]
        files.extend(input_images)
    return files
        

class LolDatasetLoader(Dataset):
    def __init__(self, flare: bool, transform = None):
        self.flare = flare
        self.inputs = []
        self.targets = []
        self.transform = transform
        self.collect_images()

    def collect_images(self):
        included_extenstions = ['png']

        if self.flare:
            self.inputs_dirs = [r'Data/LOLdataset/our485/low']
            self.targets_dirs = [r'Data/LOLdataset/our485/high']
            scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
            self.flare_image_loader=Flare_Image_Loader(transform_base=None,transform_flare=None)
            self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)

            self.inputs = []
            self.inputs.extend(get_images(self.inputs_dirs, included_extenstions))
            self.targets = []
            self.targets.extend(get_images(self.targets_dirs, included_extenstions))
   
    def __len__(self): 
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_path = self.inputs[idx]
        target_path = self.targets[idx]
        input_image = Image.open(input_path)
        if not self.flare:
            input_image = F.to_tensor(input_image)
        target_image = Image.open(target_path)
        target_image = F.to_tensor(target_image)

        if self.flare: 
            _,_,input_image,_,flare=self.flare_image_loader.apply_flare(input_image)

            input_image = input_image.transpose(1, 2)
            if self.transform:
                input_image, target_image = identical_transform(self.transform, input_image, target_image)

        else:
            if self.transform:
                input_image, target_image = identical_transform(self.transform, input_image, target_image)

        return input_image, target_image

class LolTestDatasetLoader(LolDatasetLoader):
    def __init__(self, flare: bool, transform = None):
        self.flare = flare
        self.inputs = []
        self.targets = []
        self.LensFlareLowLight = False
        self.LowLightLensFlare = False
        self.inputs_dirs = [r'Data/LOLdataset/eval15/low', r"Data/LOL-v2/Real_captured/Test/Low"]
        self.targets_dirs = [r'Data/LOLdataset/eval15/high', r"Data/LOL-v2/Real_captured/Test/Normal"]
        self.transform = transform
        scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
        self.flare_image_loader=Flare_Image_Loader(transform_base=None,transform_flare=None)
        self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)
        included_extenstions = ['png']
        self.inputs.extend(get_images(self.inputs_dirs, included_extenstions))
        self.targets.extend(get_images(self.targets_dirs, included_extenstions))

class LolValidationDatasetLoader(LolDatasetLoader):
    def __init__(self, flare: bool, transform = None):
        self.flare = flare
        self.inputs = []
        self.targets = []
        self.LensFlareLowLight = False
        self.LowLightLensFlare = False
        self.inputs_dirs = [r'Data/LOLdataset/eval15/low']
        self.targets_dirs = [r'Data/LOLdataset/eval15/high']
        self.transform = transform
        scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
        self.flare_image_loader=Flare_Image_Loader(transform_base=None,transform_flare=None)
        self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)
        included_extenstions = ['png']
        self.inputs.extend(get_images(self.inputs_dirs, included_extenstions))
        self.targets.extend(get_images(self.targets_dirs, included_extenstions))

if __name__ == "__main__":
    transform = resize_pipeline(512)
    l = LolDatasetLoader(flare=True, transform=transform)
    # l = LolTestDatasetLoader(flare=True, transform=transform)
    train_loader = DataLoader(l, batch_size=1, shuffle=True)
    i = 0 
    for input, target in train_loader:
        i += 1
        input = input.squeeze(0)
        target = target.squeeze(0)
        input_image = input.permute(1, 2, 0).detach().cpu().numpy()
        target_image = target.permute(1, 2, 0).detach().cpu().numpy()

        # if i >= 2070:
        plt.imshow(input_image)
        plt.show()
        plt.imshow(target_image)
        plt.show()
        
import sys
import os
root_dir = os.getcwd()
sys.path.append(root_dir)
import torchvision.transforms.functional as F
from Models.src.APPLY_FLARE_TO_IMG import Flare_Image_Loader
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from Modules.Preprocessing.preprocessing import resize_pipeline
import re


class SeperateDatasets(Dataset):
    def __init__(self, dataset_name, transform):
        self.transform = transform
        self.dataset = dataset_name
        self.inputs = []
        self.targets = []
        self.included_extenstions = ['png']
        self.collect_images()

    def get_images(self, dir, included_extenstions):
        files = []
        for d in dir:
            input_images = [os.path.join(d, file) for file in os.listdir(d) if any(file.endswith(ext) for ext in included_extenstions)]
            files.extend(input_images)
        return files

    def collect_images(self):
        if self.dataset == 'LensFlare':
            self.input_dirs = [r'Data/LOLdataset/our485/high']
            self.target_dirs = [r'Data/LOLdataset/our485/high']
            scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
            self.flare_image_loader = Flare_Image_Loader(transform_base=None,transform_flare=None)
            self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)

        if self.dataset == 'LowLight':
            self.input_dirs = [r'Data/LOLdataset/our485/low']
            self.target_dirs = [r'Data/LOLdataset/our485/high']

        self.inputs.extend(self.get_images(self.input_dirs, self.included_extenstions))
        self.targets.extend(self.get_images(self.target_dirs, self.included_extenstions))

        if self.dataset == 'LensFlare':
            self.inputs = sorted(self.inputs)
            self.targets = sorted(self.targets)
            
    def __len__(self): 
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_path = self.inputs[idx]
        target_path = self.targets[idx]
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        if self.dataset == 'LensFlare':
            if 'Flare7Kpp' not in input_path and 'How to Train Neural' not in input_path:
                _,_,input_image,_,_=self.flare_image_loader.apply_flare(input_image)
                input_image = input_image.transpose(1, 2)
                target_image = F.to_tensor(target_image)
            else:
                input_image = F.to_tensor(input_image)
                target_image = F.to_tensor(target_image)

        if self.dataset == 'LowLight':
            input_image = F.to_tensor(input_image)
            target_image = F.to_tensor(target_image)
        
        input_image, target_image = self.identical_transform(self.transform, input_image, target_image)

        return input_image, target_image

    def identical_transform(self, transform, input, target):
        concat = torch.cat((input.unsqueeze(0), target.unsqueeze(0)),0)
        transformed = transform(concat)
        input_trans = transformed[0]
        target_trans = transformed[1]
        return input_trans, target_trans
    
class SeperateDatasetsValidation(SeperateDatasets):
    def __init__(self, dataset_name, transform):
        super().__init__(dataset_name, transform)

    def collect_images(self):
        if self.dataset == 'LensFlare':
            self.input_dirs = [r'Data/LOLdataset/eval15/high']
            self.target_dirs = [r'Data/LOLdataset/eval15/high']
            scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
            self.flare_image_loader = Flare_Image_Loader(transform_base=None,transform_flare=None)
            self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)

        if self.dataset == 'LowLight':
            self.input_dirs = [r'Data/LOLdataset/eval15/low']
            self.target_dirs = [r'Data/LOLdataset/eval15/high']

        self.inputs.extend(self.get_images(self.input_dirs, self.included_extenstions))
        self.targets.extend(self.get_images(self.target_dirs, self.included_extenstions))

def check_sorted(inputs, targets, input_dirs, target_dirs):
    file_inputs = []
    file_targets = []
    for input_dir in input_dirs:
        for input in inputs:
            if input_dir in input:
                file_input = re.sub(input_dir, "", input)
                file_inputs.append(file_input)

    for target_dir in target_dirs:
        for target in targets:
            if target_dir in target:
                file_target = re.sub(target_dir, "", target)
                file_targets.append(file_target)

    for index, _ in enumerate(file_inputs):
        if file_inputs[index] != file_targets[index]:
            print(f"Match error: {file_inputs[index]} - {file_targets[index]}")

if __name__ == "__main__":
    transform = resize_pipeline(512)
    dataset = SeperateDatasetsValidation(dataset_name='LowLight', transform=transform)
    check_sorted(dataset.inputs, dataset.targets, dataset.input_dirs, dataset.target_dirs)
    #print(dataset.inputs)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    i = 0 
    for input, target in train_loader:
        i += 1
        input = input.squeeze(0)
        target = target.squeeze(0)
        input_image = input.permute(1, 2, 0).detach().cpu().numpy()
        target_image = target.permute(1, 2, 0).detach().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(input_image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(target_image)
        axs[1].set_title('Output Image')
        axs[1].axis('off')
        plt.show()

        
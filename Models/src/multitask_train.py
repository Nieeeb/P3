import sys
import os
root_dir = os.getcwd()
sys.path.append(root_dir)
import torchvision.transforms.functional as F
from Models.src.APPLY_FLARE_TO_IMG import Flare_Image_Loader
from PIL import Image
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
from Modules.Preprocessing.preprocessing import resize_pipeline
import re
from Models.model_zoo.U_Net import U_Net
from Models.src.CLARITY_dataloader import LolValidationDatasetLoader
import wandb


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, output, target):
        loss = torch.mean(torch.sqrt((target - output)**2 + self.epsilon**2))
        return loss

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
            if self.dataset == 'LensFlare':
                files.extend(input_images[:((len(input_images) // 2) + 1)])
            if self.dataset == 'LowLight':
                files.extend(input_images[:(len(input_images) // 2)])
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
    

transform = resize_pipeline(512)
lensflare_dataset = SeperateDatasets(dataset_name='LensFlare', transform=transform)
lensflaredl = DataLoader(lensflare_dataset, batch_size=1, shuffle=True)
lowlight_dataset = SeperateDatasets(dataset_name='LowLight', transform=transform)
lowlightdl = DataLoader(lowlight_dataset, batch_size=1, shuffle=True)
validation_dataset = LolValidationDatasetLoader(flare=True, transform=transform)
val_loader = DataLoader(validation_dataset, batch_size=1)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = U_Net(img_ch=3, output_ch=3)

#total_params = sum(p.numel() for p in model.flop())
#print(f"Number of parameters: {total_params}")

num_epochs = 50
lowlight_loss_fn = CharbonnierLoss()
criterion = CharbonnierLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
lensflare_loss_fn = CharbonnierLoss()




wandb.init(
         project="CLARITY"
     )


model.to(device)


for epoch in range(num_epochs):
    zipped_dls = zip(lowlightdl, lensflaredl)
    for j, ((lowlight_batch_X, lowlight_batch_y), (lensflare_batch_X, lensflare_batch_y)) in enumerate(zipped_dls):
        print('Training...')
        lowlight_batch_X = lowlight_batch_X.to(device)
        lowlight_batch_y = lowlight_batch_y.to(device)

        lensflare_batch_X = lensflare_batch_X.to(device)
        lensflare_batch_y = lensflare_batch_y.to(device)


        lowlight_preds = model(lowlight_batch_X)
        lowlight_loss = lowlight_loss_fn(lowlight_preds, lowlight_batch_y)
        
        lensflare_preds = model(lensflare_batch_X)
        lensflare_loss = lensflare_loss_fn(lensflare_preds, lensflare_batch_y)
        
        loss = lensflare_loss + lowlight_loss
        #print(loss.item())

        wandb.log({"Training Loss": loss})


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs) 

            loss = criterion(outputs, targets)

            #print(loss)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    wandb.log({'epoch': epoch, 'avg_val_loss': avg_val_loss})
    torch.save(model.state_dict(), f'OutputsMulti/epoch_{epoch}.pth')

torch.save(model.state_dict(), 'OutputsMulti/final.pth')

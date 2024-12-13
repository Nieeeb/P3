import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

from Models.model_zoo.CAN import CANModel
from Models.model_zoo.mirnet_v2_arch import MIRNet_v2
from Models.model_zoo.U_Net import U_Net
from Models.model_zoo.U_Net_no_skip import U_Net_no_skip
from Models.model_zoo.CIDNet import CIDNet
from Models.src.CLARITY_dataloader import LolDatasetLoader, LolValidationDatasetLoader, LolTestDatasetLoader
from Models.src.seperate_datasets import SeperateDatasets, SeperateDatasetsValidation
from Modules.Preprocessing.preprocessing import crop_flip_pipeline, cropping_only_pipeline, resize_pipeline, random_crop_and_flip_pipeline
from torch import nn
import torch
import torch.optim as optim
from scipy import ndimage
from skimage.morphology import disk
from skimage.measure import regionprops, label
import numpy as np
import matplotlib.pyplot as plt
#from torchmetrics.image import TotalVariation

class CharbonnierLossWeighted(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLossWeighted, self).__init__()
        self.epsilon = epsilon
        self.flare_gain = 0.5
    
    def forward(self, outputs, targets):
        non_flare_loss = torch.mean(torch.sqrt((targets - outputs)**2 + self.epsilon**2))
        flare_loss = calculate_flare_loss(outputs, targets, batch_size=8) 
        flare_loss = flare_loss * self.flare_gain
        return non_flare_loss + flare_loss
 
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, output, target):
        loss = torch.mean(torch.sqrt((target - output)**2 + self.epsilon**2))
        return loss

def prepare_loss(loss):
    # Define the loss function
    if loss == 'charbonnier_weighted':
        criterion = CharbonnierLossWeighted()
    elif loss == 'charbonnier':
        criterion = CharbonnierLoss()
    elif loss == 'L1':
        criterion = nn.L1Loss()
    else:
        print("Wrong loss type given. Accepted inputs: charbonnier, charbonnier_weighted, L1")
        criterion = None
    return criterion

def prepare_dataset(dataset_name, transform):
    if dataset_name == 'LensFlare':
        dataset = SeperateDatasets(dataset_name='LensFlare', transform=transform)
    elif dataset_name == 'LowLight':
        dataset = SeperateDatasets(dataset_name='LowLight', transform=transform)
    elif dataset_name == 'Mixed':
        dataset = LolDatasetLoader(flare=True, transform=transform)
    else:
        print("Wrong dataset type given. Accepted inputs: LowLight, LensFlare, Mixed")
        dataset = None
    return dataset

def prepare_default_state(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size):
    state = {
            'model': model_name,
            'optimizer': optimizer_name,
            'preprocessing_name': preprocessing_name,
            'preprocessing_size': preprocessing_size,
            'dataset': dataset_name,
            'current_epoch': 0,
            'num_epochs': 50,
            'save_interval': 5,
            'best_val_loss': 100,
            'epochs_without_improvement': 0,
            'patience': 50,
            'output_path': output_path,
            'loss': loss,
            'batch_size': batch_size,
            'scheduler': 'CosineAnnealing'
        }
    return state

def prepare_folders(model_name, preprocessing_name, dataset_name, output_path, loss):
    model_path = output_path + model_name + '_' + preprocessing_name + '_' + dataset_name + '_' + loss

    expected_folders = [output_path, model_path,
                        model_path + '/model_checkpoints/',
                        model_path + '/optimizer_checkpoints/',
                        model_path + '/training_states/',
                        model_path + '/scheduler_checkpoints/'
                        ]

    for folder in expected_folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)

def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)


def prepare_model(model_name):
    if model_name == 'MIRNet':
        model = MIRNet_v2(inp_channels=3, out_channels=3)
    elif model_name == 'UNet':
        model = U_Net(img_ch=3, output_ch=3)
    elif model_name == 'UNetNoSkip':
        model = U_Net_no_skip(img_ch=3, output_ch=3)
    elif model_name == 'CAN':
        model = CANModel(input_channels=3, out_channels=3, conv_channels=3, num_blocks=8)
    elif model_name == "CIDNet":
        model = CIDNet()
    else:
        print("Wrong model type given. Accepted inputs: MIRNet, UNet, CAN, UNetNoSkip")
        model = None
    model.apply(init_weights)
    return model

def prepare_optimizer(optimizer_name, model):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
    else:
        print("Wrong optimizer type given. Accepted inputs: Adam")
        optimizer = None
    return optimizer

def prepare_preprocessor(preprocessing_name, preprocessing_size):
    if preprocessing_name == 'crop_only':
        transform = cropping_only_pipeline(size=int(preprocessing_size))
    elif preprocessing_name == 'resize':
        transform = resize_pipeline(size=int(preprocessing_size))
    elif preprocessing_name == 'crop_flip':
        transform = crop_flip_pipeline(size=int(preprocessing_size))
    elif preprocessing_name == 'random_crop_flip':
        transform = random_crop_and_flip_pipeline(size=int(preprocessing_size))
    else:
        print("Wrong preprocessing type given. Accepted inputs: crop_only, resize, crop_flip, random_crop_flip")
        transform = None
    return transform

def load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    prepare_folders(model_name, preprocessing_name, dataset_name, output_path, loss)
    states_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/training_states/"
    states = os.listdir(states_path)

    if len(states) >= 1:

        highest_epoch = 0
        current_state = None
        for state in states:
            epoch = int(state.split('_')[3].split('.')[0])
            if epoch >= highest_epoch:
                highest_epoch = epoch
                current_state = state
        state_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/training_states/{current_state}"
        state = torch.load(state_path, weights_only=False)
        prepare_folders(model_name, preprocessing_name, dataset_name, output_path, loss)

        model_name = state['model']
        model_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/model_checkpoints/{model_name}_model_epoch_{highest_epoch}.pth"
        model = prepare_model(model_name)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.to(device)

        optimizer_name = state['optimizer']
        optimizer_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/optimizer_checkpoints/{model_name}_{optimizer_name}_optimizer_epoch_{highest_epoch}.pth"
        optimizer = prepare_optimizer(optimizer_name, model)
        optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))

        scheduler_name = state['scheduler']
        scheduler_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/scheduler_checkpoints/{model_name}_{scheduler_name}_optimizer_epoch_{highest_epoch}.pth"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, state['num_epochs'], last_epoch=state['current_epoch']-1, eta_min=1e-6)
        scheduler.load_state_dict(torch.load(scheduler_path, weights_only=False))


    else:
        model = prepare_model(model_name)
        model.to(device)
        optimizer = prepare_optimizer(optimizer_name, model)
        state = prepare_default_state(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, state['num_epochs'], last_epoch=state['current_epoch']-1, eta_min=1e-6)

    return model, optimizer, scheduler, state

def save_model(model, optimizer, scheduler, state):
    current_epoch = state['current_epoch']
    output_path = state['output_path']
    model_name = state['model']
    optimizer_name = state['optimizer']
    preprocessing_name = state['preprocessing_name']
    preprocessing_size = state['preprocessing_size']
    dataset_name = state['dataset']
    loss_name = state['loss']
    scheduler_name = state['scheduler']
    prepare_folders(model_name, preprocessing_name, dataset_name, output_path, loss_name)
    
    model_save_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss_name}/model_checkpoints/{model_name}_model_epoch_{current_epoch}.pth"
    torch.save(model.state_dict(), model_save_path)
    optimizer_save_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss_name}/optimizer_checkpoints/{model_name}_{optimizer_name}_optimizer_epoch_{current_epoch}.pth"
    torch.save(optimizer.state_dict(), optimizer_save_path)
    state_save_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss_name}/training_states/{model_name}_state_epoch_{current_epoch}.pth"
    torch.save(state, state_save_path)
    scheduler_save_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss_name}/scheduler_checkpoints/{model_name}_{scheduler_name}_optimizer_epoch_{current_epoch}.pth"
    torch.save(scheduler.state_dict(), scheduler_save_path)

def plot_light_pos(input_img,threshold):
	#input should be a three channel tensor with shape [C,H,W]
	#Out put the position (x,y) in int
     
    input_img = input_img.squeeze(0)
    luminance=0.3*input_img[0]+0.59*input_img[1]+0.11*input_img[2] # her beregner den luminance af billedet baseret på luminance equation som har vægte for hvor meget mennesker ser hver farve
    luminance_mask=luminance>threshold # Den her sat et threshold og for pixel der overskrider der laver den en luminance mask
    luminance_mask_np=luminance_mask.cpu().numpy() # Den gør masken til numpy
    struc = disk(3) #
    img_e = ndimage.binary_erosion(luminance_mask_np, structure = struc)
    img_ed = ndimage.binary_dilation(img_e, structure = struc) 
    labels = label(img_ed)

    if labels.max() == 0:
        return None

    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        largestCC=largestCC.astype(int)
        properties = regionprops(largestCC, largestCC)
        weighted_center_of_mass = properties[0].weighted_centroid
        light_pos = (int(weighted_center_of_mass[1]),int(weighted_center_of_mass[0]))
        light_pos=[light_pos[0],light_pos[1]]

        return light_pos
    

def calculate_flare_loss(outputs_batch, targets_batch, batch_size):
        flare_loss = []
        rect_size = 150
        epsilon = 1e-3
        for img in range(batch_size):
            light_pos = plot_light_pos(outputs_batch[img], 0.9)
            if light_pos is not None:
                x1 = int(max(light_pos[0] - rect_size / 2, 0))
                x2 = int(min(light_pos[0] + rect_size / 2, 512))
                y1 = int(max(light_pos[1] - rect_size / 2, 0))
                y2 = int(min(light_pos[1] + rect_size / 2, 512))
                output = outputs_batch[img].squeeze(0).permute(1, 2, 0)
                target = targets_batch[img].squeeze(0).permute(1, 2, 0)
                output = output[y1:y2, x1:x2 :]
                target = target[y1:y2, x1:x2 :]
                flare_loss.append(torch.mean(torch.sqrt((target - output)**2 + epsilon**2)))
            else:
                continue
        sum = 0
        for loss in flare_loss:
            sum += loss
            res = sum / len(flare_loss)
        return res






if __name__ == "__main__":
    state = prepare_default_state('MIRNet', 'Adam', 'crop_only', 512, 'Mixed', 'Outputs/', 'charbonnier', 8)
    model = MIRNet_v2(inp_channels=3, out_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, state['num_epochs'], last_epoch=state['current_epoch']-1, eta_min=1e-6)

    save_model(model, optimizer, scheduler, state)
    preprocessing_name = 'crop_only'
    preprocessing_size = 512
    dataset_name = 'Mixed'
    loss = 'charbonnier'
    model, optimizer, scheduler, state = load_latest_checkpoint('MIRNet', 'Adam', preprocessing_name, preprocessing_size, dataset_name, 'Outputs/', loss, 8)

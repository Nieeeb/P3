import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

from Models.model_zoo.CAN import CANModel
from Models.model_zoo.mirnet_v2_arch import MIRNet_v2
from Models.model_zoo.U_Net import U_Net
from Models.src.CLARITY_dataloader import LolDatasetLoader, LolValidationDatasetLoader, LolTestDatasetLoader

from Preprocessing.preprocessing import crop_flip_pipeline, cropping_only_pipeline, resize_pipeline, random_crop_and_flip_pipeline
from torch import nn
import torch
import torch.optim as optim
from torchmetrics.image import TotalVariation

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, output, target):
        return torch.mean(torch.sqrt((target - output)**2 + self.epsilon**2))

def prepare_loss(loss):
    # Define the loss function
    if loss == 'charbonnier':
        criterion = CharbonnierLoss()
    elif loss == 'total_variation':
        criterion = TotalVariation()
    else:
        print("Wrong loss type given. Accepted inputs: charbonnier, total_variation")
        criterion = None
    return criterion

def prepare_dataset(dataset_name, transform):
    if dataset_name == 'LowLightLensFlare':
        dataset = LolDatasetLoader(flare=False, LowLightLensFlare=True, LensFlareLowLight=False, transform=transform)
    elif dataset_name == 'LensFlareLowLight':
        dataset = LolDatasetLoader(flare=False, LowLightLensFlare=False, LensFlareLowLight=True, transform=transform)
    elif dataset_name == 'Mixed':
        dataset = LolDatasetLoader(flare=True, LowLightLensFlare=False, LensFlareLowLight=False, transform=transform)
    else:
        print("Wrong dataset type given. Accepted inputs: LowLightLensFlare, LensFlareLowLight, Mixed")
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
            'num_epochs': 1000,
            'save_interval': 10,
            'best_val_loss': 100,
            'epochs_without_improvement': 0,
            'patience': 10,
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
    elif model_name == 'CAN':
        model = CANModel(input_channels=3, out_channels=3, conv_channels=3, num_blocks=8)
    else:
        print("Wrong model type given. Accepted inputs: MIRNet, UNet, CAN")
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

def load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size):
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
        model.load_state_dict(torch.load(model_path, weights_only=True))

        optimizer_name = state['optimizer']
        optimizer_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/optimizer_checkpoints/{model_name}_{optimizer_name}_optimizer_epoch_{highest_epoch}.pth"
        optimizer = prepare_optimizer(optimizer_name, model)
        optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

        scheduler_name = state['scheduler']
        scheduler_path = f"{output_path}{model_name}_{preprocessing_name}_{dataset_name}_{loss}/scheduler_checkpoints/{model_name}_{scheduler_name}_optimizer_epoch_{highest_epoch}.pth"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, state['num_epochs'], last_epoch=state['current_epoch']-1, eta_min=1e-6)
        scheduler.load_state_dict(torch.load(scheduler_path, weights_only=False))


    else:
        model = prepare_model(model_name)
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

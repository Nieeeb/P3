import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

from Models.model_zoo.CAN import CANModel
from Models.model_zoo.mirnet_v2_arch import MIRNet_v2
from Models.model_zoo.U_Net import U_Net

from Preprocessing.preprocessing import crop_flip_pipeline, cropping_only_pipeline, resize_pipeline, random_crop_and_flip_pipeline

import torch
import torch.optim as optim

def prepare_default_state(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path):
    state = {
            'model': model_name,
            'optimizer': optimizer_name,
            'preprocessing_name': preprocessing_name,
            'preprocessing_size': preprocessing_size,
            'current_epoch': 0,
            'num_epochs': 1000,
            'save_interval': 10,
            'best_val_loss': 100,
            'epochs_without_improvement': 0,
            'patience': 10,
            'output_path': output_path
        }
    return state

def prepare_folders(model_name, preprocessing_name, output_path):
    model_path = output_path + model_name + '_' + preprocessing_name

    expected_folders = [output_path, model_path,
                        model_path + '/model_checkpoints/',
                        model_path + '/optimizer_checkpoints/',
                        model_path + '/training_states/'
                        ]

    for folder in expected_folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)

def prepare_model(model_name):
    if model_name == 'MIRNet':
        model = MIRNet_v2(inp_channels=3, out_channels=3)
    elif model_name == 'UNet':
        model = U_Net(img_ch=3, output_ch=3)
    elif model_name == 'CAN':
        model = CANModel(input_channels=3, out_channels=3, conv_channels=3, num_blocks=1)
    else:
        print("Wrong model type given. Accepted inputs: MIRNet, UNet, CAN")
        model = None
    return model

def prepare_optimizer(optimizer_name, model):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
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

def load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path):
    prepare_folders(model_name, preprocessing_name, output_path)
    states_path = f"{output_path}{model_name}_{preprocessing_name}/training_states/"
    states = os.listdir(states_path)

    if len(states) >= 1:

        highest_epoch = 0
        current_state = None
        for state in states:
            epoch = int(state.split('_')[3].split('.')[0])
            if epoch >= highest_epoch:
                highest_epoch = epoch
                current_state = state
        state_path = f"{output_path}{model_name}_{preprocessing_name}/training_states/{current_state}"
        state = torch.load(state_path, weights_only=False)
        prepare_folders(model_name, preprocessing_name, output_path)

        model_name = state['model']
        model_path = f"{output_path}{model_name}_{preprocessing_name}/model_checkpoints/{model_name}_model_epoch_{highest_epoch}.pth"
        model = prepare_model(model_name)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        optimizer_name = state['optimizer']
        optimizer_path = f"{output_path}{model_name}_{preprocessing_name}/optimizer_checkpoints/{model_name}_{optimizer_name}_optimizer_epoch_{highest_epoch}.pth"
        optimizer = prepare_optimizer(optimizer_name, model)
        optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

    else:
        model = prepare_model(model_name)
        optimizer = prepare_optimizer(optimizer_name, model)
        state = prepare_default_state(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path)
    
    return model, optimizer, state

def save_model(model, optimizer, state):
    current_epoch = state['current_epoch']
    output_path = state['output_path']
    model_name = state['model']
    optimizer_name = state['optimizer']
    preprocessing_name = state['preprocessing_name']
    preprocessing_size = state['preprocessing_size']
    prepare_folders(model_name, preprocessing_name, output_path)
    
    model_save_path = f"{output_path}{model_name}_{preprocessing_name}/model_checkpoints/{model_name}_model_epoch_{current_epoch}.pth"
    torch.save(model.state_dict(), model_save_path)
    optimizer_save_path = f"{output_path}{model_name}_{preprocessing_name}/optimizer_checkpoints/{model_name}_{optimizer_name}_optimizer_epoch_{current_epoch}.pth"
    torch.save(optimizer.state_dict(), optimizer_save_path)
    state_save_path = f"{output_path}{model_name}_{preprocessing_name}/training_states/{model_name}_state_epoch_{current_epoch}.pth"
    torch.save(state, state_save_path)

if __name__ == "__main__":
    state = prepare_default_state('MIRNet', 'Adam', 'crop_only', 512, 'Outputs/')
    model = MIRNet_v2(inp_channels=3, out_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_model(model, optimizer, state)
    preprocessing_name = 'crop_only'
    preprocessing_size = 512
    model, optimizer, state = load_latest_checkpoint('MIRNet', 'Adam', preprocessing_name, preprocessing_size, 'Outputs/')

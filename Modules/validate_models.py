import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to sys.path to ensure modules can be found
root_dir = os.getcwd()
sys.path.append(root_dir)

# Import the U_Net model class
from Models.model_zoo.U_Net import U_Net
from Models.model_zoo.CAN import CANModel
# Import the test dataset loader and any required preprocessing pipelines
from Models.src.CLARITY_dataloader import LolTestDatasetLoader, LolValidationDatasetLoader
from Modules.Preprocessing.preprocessing import crop_flip_pipeline, resize_pipeline
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Modules.checkpointing import prepare_model, load_latest_checkpoint
import pickle
#import lpips

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def validate_model(modeldict, test_loader):
    model = modeldict['model']
    model.eval()
    # Lists to store metric values
    ssim_values = []
    psnr_values = []
    lpips_values = []

    # Testing loop
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            # Move data to the appropriate device
            #inputs = inputs.type(torch.int32)
            #targets = targets.type(torch.int32)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            #outputs = outputs.squeeze(0)
            #outputs = outputs.type(torch.int32)
            # Convert predictions and targets to numpy arrays for SSIM/PSNR
            pred_np = outputs.squeeze(0).cpu().numpy()
            target_np = targets.squeeze(0).cpu().numpy()
            #print(outputs.shape)
            outputs= outputs.cpu()
            targets = targets.cpu()
            outputs = outputs.numpy()
            targets = targets.numpy()
            #print(targets)
            # Compute SSIM
            #print(target_np.shape)
            range = pred_np.max() - pred_np.min()
            ssim_val = ssim(target_np, pred_np, multi_channel=True, channel_axis=0, data_range=1.0)
            #print(ssim_val)
            ssim_values.append(ssim_val)

            # Compute PSNR
            psnr_val = compare_psnr(targets, outputs, data_range=1.0)
            psnr_values.append(psnr_val)

            # Compute LPIPS (LPIPS returns a tensor)
            #lpips_val = lpips_fn(outputs, targets)
            #lpips_values.append(lpips_val.item())

    # After the loop, compute the average of each metric
    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    #mean_lpips = np.mean(lpips_values)
    print('---------------------------------------------------------------------------')
    print(f"Results for: {modeldict['modelname']}")
    print(f"Average SSIM: {mean_ssim:.4f}")
    print(f"Average PSNR: {mean_psnr:.4f} dB")
    print(f"State:\n{modeldict['state']}")
    #print(f"Average LPIPS: {mean_lpips:.4f}")
    print('---------------------------------------------------------------------------')


datasets = ['Mixed']

augmentations = ['crop_flip', 'resize', 'crop_only']

model_names = ['UNet', 'CAN', 'CIDNet']

modeldicts = []

# Initialize the model and load the saved state dictionary
canmodel = prepare_model("CAN")
canmodel.load_state_dict(torch.load(r'/ceph/project/CLARITY/Outputs/CAN_resize_Mixed_charbonnier/model_checkpoints/CAN_model_epoch_19.pth', map_location=device))
candict = {'model': canmodel, 'modelname': 'CAN'}
modeldicts.append(candict)

unetmodel = prepare_model("UNet")
unetmodel.load_state_dict(torch.load(r'/ceph/project/CLARITY/Outputs/CAN_resize_Mixed_charbonnier/model_checkpoints/CAN_model_epoch_19.pth', map_location=device))
unetdict = {'model': unetmodel, 'modelname': 'unet'}
modeldicts.append(unetdict)

cidmodel = prepare_model("UNet")
cidmodel.load_state_dict(torch.load(r'/ceph/project/CLARITY/Outputs/CAN_resize_Mixed_charbonnier/model_checkpoints/CAN_model_epoch_19.pth', map_location=device))
ciddict = {'model': cidmodel, 'modelname': 'cidnet'}
modeldicts.append(ciddict)


# Load the test dataset
test_dataset = LolValidationDatasetLoader(flare=True, transform=resize_pipeline(512))
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

for modeldict in modeldicts:
    validate_model(modeldict, test_loader)

#lpips_fn = lpips.LPIPS(net='alex').to(device)
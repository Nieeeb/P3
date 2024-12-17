import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)
from thop import profile
from Models.src.CLARITY_dataloader import LolValidationDatasetLoader
from torch.utils.data import DataLoader
from Modules.checkpointing import prepare_preprocessor, prepare_model
from Models.model_zoo.seqentialmodel import SequentialModel
import torch

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tranform = prepare_preprocessor('resize', 512)
dataset = LolValidationDatasetLoader(flare=True, transform=tranform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
input, target = next(iter(dataloader))
input, target = input.to(device), target.to(device)


best_model = prepare_model('UNet')
best_model.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LensFlare_charbonnier\model_checkpoints\UNet_model_epoch_49LFNotDark.pth", map_location=device, weights_only=False))

model_flare = prepare_model('UNet')
model_flare.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LensFlare_charbonnier\model_checkpoints\UNet_model_epoch_49LFNotDark.pth", map_location=device, weights_only=False))

model_low = prepare_model('UNet')
model_low.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LensFlare_charbonnier\model_checkpoints\UNet_model_epoch_49LFNotDark.pth", map_location=device, weights_only=False))

sequential = SequentialModel(model_low, model_flare)

macs_best, params_best = profile(best_model, inputs=(input, ))
macs_sequential, params_sequntial = profile(sequential, inputs=(input, ))

print('---------------------------------------------------')
print(f"Our model:\nmacs: {macs_best}\nparams: {params_best}")
print('---------------------------------------------------')
print(f"Sequential model:\nmacs: {macs_sequential}\nparams: {params_sequntial}")
print('---------------------------------------------------')
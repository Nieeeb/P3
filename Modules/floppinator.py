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
from calflops import calculate_flops

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(f"Using device: {device}")

tranform = prepare_preprocessor('resize', 512)
dataset = LolValidationDatasetLoader(flare=True, transform=tranform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
input, target = next(iter(dataloader))
input, target = input.to(device), target.to(device)

#input, target = input / 255, target / 255


best_model = prepare_model('UNet')
best_model.load_state_dict(torch.load(r"/home/nieb/Projects/DAKI Projects/P3/Outputs/Models/BEST MODEL/LPIPSONLY.pth", map_location=device, weights_only=False))
best_model.to(device)

model_flare = prepare_model('UNet')
model_flare.load_state_dict(torch.load(r"Outputs/Models/flareLightLpips.pth", map_location=device, weights_only=False))

model_low = prepare_model('UNet')
model_low.load_state_dict(torch.load(r"Outputs/Models/LowLightLpips.pth", map_location=device, weights_only=False))

sequential = SequentialModel(model_low, model_flare)
sequential.to(device)

#macs_best, params_best = profile(best_model, inputs=(input, ))
#macs_sequential, params_sequntial = profile(sequential, inputs=(input, ))
flops_best, macs_best, params_best = calculate_flops(model=best_model, input_shape=tuple(input.shape), output_as_string=True, output_precision=4)
flops_sequntial, macs_sequential, params_sequntial = calculate_flops(sequential, input_shape=tuple(input.shape), output_as_string=True, output_precision=4)


print('---------------------------------------------------')
print(f"Our model:\nFlops: {flops_best}\nmacs: {macs_best}\nparams: {params_best}")
print('---------------------------------------------------')
print(f"Sequential model:Flops: {flops_sequntial}\nmacs: {macs_sequential}\nparams: {params_sequntial}")
print('---------------------------------------------------')
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
#from Models.model_zoo.U_Net import U_Net
#from Models.model_zoo.CAN import CANModel
# Import the test dataset loader and any required preprocessing pipelines
from Models.src.CLARITY_dataloader import LolTestDatasetLoader, LolDatasetLoader, LolValidationDatasetLoader
from Modules.Preprocessing.preprocessing import crop_flip_pipeline, resize_pipeline
from Modules.checkpointing import prepare_model, load_latest_checkpoint, prepare_preprocessor

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

datasets = ['Mixed']

augmentations = ['crop_flip', 'resize', 'crop_only']

model_names = ['UNet', 'CAN', 'CIDNet']

modeldicts = []

for dataset in datasets:
    for augemntation in augmentations:
        for model_name in model_names:
            model, _, _, state = load_latest_checkpoint(model_name=model_name,
                                           optimizer_name='Adam',
                                           preprocessing_name=augemntation,
                                           preprocessing_size=512,
                                           dataset_name=dataset,
                                           output_path='Outputs/Models/',
                                           loss='charbonnier',
                                           batch_size=1,
                                           device=device)
            model_name = f"{model_name}_{augemntation}_{dataset}"
            model_dict = {'model': model, 'modelname': model_name, 'state': state, 'transform': augemntation}
            modeldicts.append(model_dict)

def generate_image(modeldict, input, target):
    # Testing loop
    with torch.no_grad():
        model = modeldict['model']
        model.eval()
        # Move data to the appropriate device
        input = input.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(input)
        #print(outputs)

        # Convert tensors to numpy arrays for matplotlib
        # Remove the batch dimension and permute dimensions to (H, W, C)
        input_image = input.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = outputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
        target_image = target.cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Clip values to [0, 1] if necessary
        input_image = np.clip(input_image, 0, 1)
        output_image = np.clip(output_image, 0, 1)
        target_image = np.clip(target_image, 0, 1)

        # Display images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(input_image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(output_image)
        axs[1].set_title('Output Image')
        axs[1].axis('off')

        axs[2].imshow(target_image)
        axs[2].set_title('Target Image')
        axs[2].axis('off')

        #plt.show()

        plt.savefig(f"Outputs/ValidationImages/{modeldict['modelname']}.jpg")
        #plt.imsave(f"Outputs/ValidationImages/{modeldict['modelname']}.jpg", axs)


for modeldict in modeldicts:
    tranform = prepare_preprocessor(modeldict['transform'], 512)
    dataset = LolValidationDatasetLoader(flare=True, transform=tranform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    input, target = next(iter(dataloader))
    generate_image(modeldict, input, target)


print("Images written")

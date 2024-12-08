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
from Modules.checkpointing import prepare_model

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model and load the saved state dictionary
#model = CANModel(input_channels=3, out_channels=3, conv_channels=3, num_blocks=8)
model = prepare_model('UNet')
model.load_state_dict(torch.load('Outputs/UNet_model_epoch_13.pth'))
model.to(device)
model.eval()  # Set model to evaluation mode

# Load the test dataset
test_dataset = LolValidationDatasetLoader(flare=True, transform=resize_pipeline(512))
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Define the loss function for evaluation
criterion = nn.L1Loss()

# Initialize variables to track the test loss
test_loss = 0.0

# Testing loop
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        # Move data to the appropriate device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Convert tensors to numpy arrays for matplotlib
        # Remove the batch dimension and permute dimensions to (H, W, C)
        input_image = inputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = outputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
        target_image = targets.cpu().squeeze(0).permute(1, 2, 0).numpy()

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

        plt.show()
        plt.close()

        # Optionally, print progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_loader)} images.")

# Calculate the average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

print("Testing completed.")

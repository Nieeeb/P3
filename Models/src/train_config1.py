import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

import torch.utils
root_dir = os.getcwd()
sys.path.append(root_dir)
<<<<<<< HEAD
#from Models.model_zoo.konfig_1_Uformer_cross import Uformer_Cross
=======
>>>>>>> d103edf7 (ælkoæk)
from Models.model_zoo.U_Net import U_Net

import math
from CLARITY_dataloader import LolDatasetLoader, LolValidationDatasetLoader
from torch.utils.data import DataLoader
from Modules.Preprocessing.preprocessing import preprocessing_pipeline_example, crop_flip_pipeline, cropping_only_pipeline, random_crop_and_flip_pipeline
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = U_Net(img_ch=3, output_ch=3)
model.to(device)

# Define the loss function and optimizer
criterion = nn.L1Loss()  # L1 Loss is common for image restoration tasks
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

# Number of epochs to train
num_epochs = 1

train_dataset = LolDatasetLoader(flare=True, transform=crop_flip_pipeline(128))
<<<<<<< HEAD
train_loader = DataLoader(dataset=train_dataset, batch_size=4)

val_dataset = LolValidationDatasetLoader(flare=True, transform=crop_flip_pipeline(128))
val_loader = DataLoader(dataset=val_dataset, batch_size=4)
=======
train_loader = DataLoader(dataset=train_dataset, batch_size=8)

val_dataset = LolValidationDatasetLoader(flare=True, transform=crop_flip_pipeline(128))
val_loader = DataLoader(dataset=val_dataset, batch_size=8)
>>>>>>> d103edf7 (ælkoæk)



# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(batch_idx)
        # print(inputs.shape)
        # Move data to the appropriate device (CPU or GPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        print(loss.item())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print statistics every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 100:.4f}, "
                  f"Learning rate: {optimizer.param_groups[0]['lr']}")
            running_loss = 0.0

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Optionally save the model checkpoint
    # torch.save(model.state_dict(), f"uformer_epoch_{epoch+1}.pth")
torch.save(model.state_dict(), 'model_final.pth')

print("Training completed.")

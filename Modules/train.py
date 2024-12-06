import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)
from Models.model_zoo.CAN import CANModel
import torch
import torch.nn as nn
import torch.optim as optim
import math
from Models.src.CLARITY_dataloader import LolDatasetLoader, LolValidationDatasetLoader
from torch.utils.data import DataLoader
from Modules.Preprocessing.preprocessing import preprocessing_pipeline_example, crop_flip_pipeline, cropping_only_pipeline, random_crop_and_flip_pipeline
from tqdm import tqdm
from checkpointing import load_latest_checkpoint, save_model, prepare_preprocessor

def train(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path):
    model, optimizer, state = load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path)
    transform = prepare_preprocessor(preprocessing_name, preprocessing_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)

    # Define the loss function
    criterion = nn.L1Loss()  # L1 Loss is common for image restoration tasks

    # Number of epochs to train
    num_epochs = state['num_epochs']
    starting_epoch = state['current_epoch']
    save_interval = state['save_interval']

    train_dataset = LolDatasetLoader(flare=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=2)

    val_dataset = LolValidationDatasetLoader(flare=True, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2)

    patience = state['patience']

    best_val_loss = state['best_val_loss']

    # Training loop
    for epoch in range(starting_epoch, num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        state['current_epoch'] = epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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


        #EARLY DROPOUT
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state['bestbest_val_loss'] = best_val_loss
            epochs_without_improvement = 0
            state['epochs_without_improvement'] = epochs_without_improvement
            # Save the best model
            save_model(model, optimizer, state)
            print(f"Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            state['epochs_without_improvement'] = epochs_without_improvement
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        if epoch // save_interval == 0:
            save_model(model, optimizer, state)
        # Optionally save the model checkpoint
        # torch.save(model.state_dict(), f"uformer_epoch_{epoch+1}.pth")

    print("Training completed.")

if __name__ == "__main__":
    model = 'MIRNet'
    optimizer = 'Adam'
    preprocessing_name = 'crop_only'
    preprocessing_size = 512
    output_path = 'Outputs/'
    train(model, optimizer, preprocessing_name, preprocessing_size, output_path)

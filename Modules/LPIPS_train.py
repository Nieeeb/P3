import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
from Models.src.CLARITY_dataloader import LolDatasetLoader, LolValidationDatasetLoader
from Models.src.seperate_datasets import SeperateDatasets, SeperateDatasetsValidation
from torch.utils.data import DataLoader
from tqdm import tqdm
from checkpointing import load_latest_checkpoint, save_model, prepare_preprocessor, prepare_loss, prepare_dataset
import wandb
import argparse
import random
import numpy as np
from scipy import ndimage
from skimage.morphology import disk
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import lpips

def train(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, optimizer, scheduler, state = load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size, device)
    transform = prepare_preprocessor(preprocessing_name, preprocessing_size)
    wandb.login()

    wandb.init(
        project="CLARITY",
        config=state,
    )

    print("Loaded state:\n", state)

    model.to(device)

    loss_fn_LPIPS = lpips.LPIPS(net='alex')
    loss_fn_LPIPS = loss_fn_LPIPS.to(device)
    loss_fn_distortion = prepare_loss(state['loss'])

    # Number of epochs to train
    num_epochs = state['num_epochs']
    starting_epoch = state['current_epoch']
    save_interval = state['save_interval']

    dataset_name = state['dataset']
    batch_size = state['batch_size']
    train_dataset = prepare_dataset(dataset_name, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    if dataset_name == 'Mixed':
        val_dataset = LolValidationDatasetLoader(flare=True, transform=transform)
    else:
        val_dataset = SeperateDatasetsValidation(dataset_name=dataset_name, transform=transform)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    patience = state['patience']

    best_val_loss = state['best_val_loss']

    epochs_without_improvement = state['epochs_without_improvement']

    # Training loop
    for epoch in range(starting_epoch, num_epochs):
        print(f"Starting training for epoch {epoch}")
        model.train()  # Set model to training mode
        running_loss = 0.0

        state['current_epoch'] = epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader):        
            # Move data to the appropriate device (CPU or GPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss_LPIPS = loss_fn_LPIPS(outputs, targets)
            loss_LPIPS = torch.mean(loss_LPIPS)
            loss_distortion = loss_fn_distortion(outputs, targets)

            combined_loss = loss_LPIPS + loss_distortion


            # Compute loss
            wandb.log({"Training Loss": combined_loss})
            
            # Backward pass and optimize
            combined_loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += combined_loss.item()

            print(running_loss)

            # Print statistics every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 100:.4f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}")
                running_loss = 0.0

        save_model(model=model, optimizer=optimizer, scheduler=scheduler, state=state)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs) 

                loss_LPIPS = loss_fn_LPIPS(outputs, targets)
                loss_LPIPS = torch.mean(loss_LPIPS)

                loss = loss_LPIPS 
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        scheduler.step()
        wandb.log({'epoch': epoch, 'avg_val_loss': avg_val_loss, 'lr': scheduler.get_last_lr()})

        #EARLY DROPOUT
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state['bestbest_val_loss'] = best_val_loss
            epochs_without_improvement = 0
            state['epochs_without_improvement'] = epochs_without_improvement
            # Save the best model
            save_model(model, optimizer, scheduler, state)
            print(f"Validation loss improved. Model saved.")
        else:
            epochs_without_improvement += 1
            state['epochs_without_improvement'] = epochs_without_improvement
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                save_model(model, optimizer, scheduler, state)
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        if epoch // save_interval == 0:
            save_model(model, optimizer, scheduler, state)
        # Optionally save the model checkpoint
        # torch.save(model.state_dict(), f"uformer_epoch_{epoch+1}.pth")
    save_model(model, optimizer, scheduler, state)
    print("Training completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--loss', type=str, choices=['charbonnier', 'charbonnier_weighted', 'L1', 'LPIPS'], default='charbonnier_weighted', help="Loss function")
    parser.add_argument('--model', type=str, choices=['MIRNet', 'UNet', 'CAN', "CIDNet", "UnetNoSkip"], default='UNet', help="What model to train")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--preprocessing_name', type=str, choices=['crop_only', 'resize', 'crop_flip', 'random_crop_flip'], default='resize', help="How to augment images")
    parser.add_argument('--preprocessing_size', type=int, default=512, help="Desired input size")
    parser.add_argument('--optimizer', type=str, choices=['Adam'], default='Adam', help="What optimizer to use")
    parser.add_argument('--scheduler', type=str, choices=['CosineAnnealing'], default='CosineAnnealing', help="Learning rate scheduler")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--dataset', type=str, choices=['Mixed', "LensFlare", "LowLight"], default='Mixed', help="What data to train on")
    parser.add_argument('--output_path', type=str, default='Outputs/', help="Where to output checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    preprocessing_name = args.preprocessing_name
    preprocessing_size = args.preprocessing_size
    dataset_name = args.dataset
    output_path = args.output_path
    loss = args.loss
    batch_size = args.batch_size

    train(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size)

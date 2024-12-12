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
from torch.utils.data import DataLoader
from tqdm import tqdm
from checkpointing import load_latest_checkpoint, save_model, prepare_preprocessor, prepare_loss, prepare_dataset
import wandb
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

def train(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, optimizer, scheduler, state = load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size, device)
    transform = prepare_preprocessor(preprocessing_name, preprocessing_size)

    print("Loaded state:\n", state)

    model.to(device)

    loss_function = state['loss']
    criterion = prepare_loss(loss_function)

    # Number of epochs to train
    num_epochs = 150
    starting_epoch = state['current_epoch']
    save_interval = state['save_interval']

    dataset_name = state['dataset']
    batch_size = state['batch_size']
    train_dataset = prepare_dataset(dataset_name, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    val_dataset = LolValidationDatasetLoader(flare=True, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    patience = state['patience']

    best_val_loss = state['best_val_loss']

    epochs_without_improvement = state['epochs_without_improvement']

    # Training loop
    for epoch in range(starting_epoch, num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        state['current_epoch'] = epoch

        # print(inputs.shape)
        # Move data to the appropriate device (CPU or GPU)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # print(outputs)
        # Compute loss
        loss = criterion(outputs, targets)
        print(f"Loss: {loss}, Epoch: {epoch}")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print statistics every 100 batches

        # Validation loop
        scheduler.step()

                # Convert tensors to numpy arrays for matplotlib
        # Remove the batch dimension and permute dimensions to (H, W, C)
        input_image = inputs.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        output_image = outputs.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        target_image = targets.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

        #output_image = input_image + output_image

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

        plt.savefig(f"Outputs/overfitting/beforebeef.jpg")

        # Optionally save the model checkpoint
        # torch.save(model.state_dict(), f"uformer_epoch_{epoch+1}.pth")
    save_model(model, optimizer, scheduler, state)
    print("Training completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--loss', type=str, choices=['charbonnier', 'total_variation', 'L1'], default='L1', help="Loss function")
    parser.add_argument('--model', type=str, choices=['MIRNet', 'UNet', 'CAN', "CIDNet"], default='UNet', help="What model to train")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--preprocessing_name', type=str, choices=['crop_only', 'resize', 'crop_flip', 'random_crop_flip'], default='resize', help="How to augment images")
    parser.add_argument('--preprocessing_size', type=int, default=512, help="Desired input size")
    parser.add_argument('--optimizer', type=str, choices=['Adam'], default='Adam', help="What optimizer to use")
    parser.add_argument('--scheduler', type=str, choices=['CosineAnnealing'], default='CosineAnnealing', help="Learning rate scheduler")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--dataset', type=str, choices=['Mixed', 'LowLightLensFlare', 'LensFlareLowLight'], default='Mixed', help="What data to train on")
    parser.add_argument('--output_path', type=str, default='Outputs/', help="Where to output checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    args = parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    preprocessing_name = args.preprocessing_name
    preprocessing_size = args.preprocessing_size
    dataset_name = args.dataset
    output_path = "Outputs/overfitting/"
    loss = 'charbonnier'
    batch_size = 1
    train(model_name, optimizer_name, preprocessing_name, preprocessing_size, dataset_name, output_path, loss, batch_size)

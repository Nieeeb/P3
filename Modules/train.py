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
from checkpointing import load_latest_checkpoint, save_model, prepare_preprocessor
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--loss', choices=['charbonnier', 'total_variation'], default='charbonnier', help="Loss function")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, choices=[4, 8], default=4, help="Batch size")
    parser.add_argument('--scheduler', choices=['cosine'], default='cosine', help="Learning rate scheduler")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate for the scheduler")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--max_epochs', type=int, default=50, help="Maximum number of epochs")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    return parser.parse_args()

def train(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path, args):
    model, optimizer, state = load_latest_checkpoint(model_name, optimizer_name, preprocessing_name, preprocessing_size, output_path)
    transform = prepare_preprocessor(preprocessing_name, preprocessing_size)

    wandb.login()

    wandb.init(
        project="CLARITY",
        config=state
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)

    # Define the loss function
    if args.loss == 'charbonnier':
        criterion = CharbonnierLoss()
    if args.loss == 'total_variation':
        criterion = TotalVariationLoss()
    criterion = nn.L1Loss()  # L1 Loss is common for image restoration tasks

    # Number of epochs to train
    num_epochs = state['num_epochs']
    starting_epoch = state['current_epoch']
    save_interval = state['save_interval']


    if state['dataset'] == 'LowLightLensFlare':
        train_dataset = LolDatasetLoader(flare=False, transform=transform, LowLightLensFlare=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)

        val_dataset = LolValidationDatasetLoader(flare=True, transform=transform, LowLightLensFlare=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=8)


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

        wandb.log({'epoch': epoch, 'avg_val_loss': avg_val_loss})


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
    dataset_name = 'wtf is this shit'
    output_path = 'Outputs/'
    args = parse_args()
    train(model, optimizer, preprocessing_name, preprocessing_size, output_path, args=args)

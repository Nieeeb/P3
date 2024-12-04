import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

import torch.utils
root_dir = os.getcwd()
sys.path.append(root_dir)
from Models.model_zoo.konfig_1_Uformer_cross import Uformer_Cross
import math
from CLARITY_dataloader import LolDatasetLoader
from torch.utils.data import DataLoader



#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

# Assuming you have already imported or defined the Uformer_Cross model class
# from your provided code.

# Check if a GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the Uformer_Cross model
model = Uformer_Cross(
    img_size=128,     # Adjust according to your image size
    in_chans=3,       # Number of input channels (e.g., 3 for RGB images)
    out_chans=3,      # Number of output channels
    embed_dim=32,     # Embedding dimension, adjust based on model size
    depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],  # Depth of each stage
    num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1],  # Number of attention heads
    win_size=8,       # Window size for attention
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=nn.LayerNorm,
    patch_norm=True,
    use_checkpoint=False,
    token_projection='linear',
    token_mlp='ffn',
    dowsample=Downsample, #er stavet forkert men det må IKKE rettes til downsample Det ville fucke scriptet op
    upsample=Upsample
).to(device)

# Define the loss function and optimizer
criterion = nn.L1Loss()  # L1 Loss is common for image restoration tasks
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

# Number of epochs to train
num_epochs = 50

dataset = LolDatasetLoader(flare=False, light_source_on_target=False)
train_loader = DataLoader(dataset=dataset)





# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
                  f"Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Optionally save the model checkpoint
    # torch.save(model.state_dict(), f"uformer_epoch_{epoch+1}.pth")

print("Training completed.")

from CLARITY_dataloader import LolDatasetLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from CLARITY_model_zoo.simplecnn import ImageToImageCNN 

dataset = LolDatasetLoader(flare=False, light_source_on_target=False)
train_loader = DataLoader(dataset=dataset)

'''
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) #Resnet outputter ikke et billede men derimod tensor shape = [1, 1000]
'''

model = ImageToImageCNN()

epochs = 1

optimizer = optim.Adam(model.parameters(), lr=0.01)

def charbonnier_loss(y_true, y_pred): #loss function brugt i MIRNET (til fordel for crossentropy som bliver brugt til klassifikation)
    epsilon = 1e-3
    return torch.mean(torch.sqrt(torch.square(y_true - y_pred) + epsilon**2)) #originalt skrevet i tensorflow men omskrevet til torch: https://keras.io/examples/vision/mirnet/ - 'Training'

for epoch in range(epochs):
    total_loss = 0
    for input, target in train_loader:

        model.train()
        optimizer.zero_grad()
        output = model(input)
        loss = charbonnier_loss(target, output)
        print(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avrg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch + 1}, Loss: {avrg_loss}')
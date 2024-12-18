import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import cv2 as cv
import torch
import numpy as np
from U_Net import U_Net
import torchvision.transforms.v2 as T
import timeit

def generate_image(input, output, tic, model_time):
    input_image = input.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

    #output_image = input_image + output_image

    # Clip values to [0, 1] if necessary
    input_image = np.clip(input_image, 0, 1)
    output_image = np.clip(output_image, 0, 1)

    #print(input_image.shape)
    #print(output_image.shape)

    # Display images
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(input_image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[0].text(1.35, -0.05, f'Model processing time {model_time}s', size=12, ha="center", transform=axs[0].transAxes)

    axs[1].imshow(output_image)
    axs[1].set_title('Output Image')
    axs[1].axis('off')
    toc = round(timeit.default_timer() - tic, 5)
    axs[1].text(-0.3, -0.1, f'End to end processing time {toc}s', size=12, ha="center", transform=axs[1].transAxes)

def tensor_to_cv(tensor_image):
    cv_image = cv.cvtColor(tensor_image.numpy().transpose(1, 2, 0), cv.COLOR_BGR2RGB)
    return cv_image

def resize_pipeline(size):
    transform = T.Compose([
        T.ToImage(),
        #FixedHeightResize(size),
        T.Resize((size, size)),
    ])
    return transform

# THis function should return a pre-trained model and the augmentation necesary
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    model = U_Net(img_ch=3, output_ch=3)
    model_path = "LPIPSONLY.pth"
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    # Disable dropout and batzh normilization
    model.eval()
    
    transform = resize_pipeline(512)

    return model, transform


# This function should apply the model with the given input and return the output
def apply_model(model, transform, input):
    model_tic = timeit.default_timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Right now it doesn't do anything, since we dont have a model
    input = transform(input)

    input = input/255

    input = input.unsqueeze(0)

    input = input.to(device)
    model.to(device)

    output = model(input)
    #prediction = model(processed_input)
    model_toc = timeit.default_timer()
    model_time = round(model_toc - model_tic, 5)
    return output, model_time

def process_image(image, model, transform, tic):
    input_tensor = transform(image)
    input_tensor = input_tensor / 255
    output_tensor, model_time = apply_model(model, transform, image)
    generate_image(input_tensor, output_tensor, tic, model_time)

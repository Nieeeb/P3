import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import torch
import numpy as np
from Modules.checkpointing import prepare_model, prepare_preprocessor

def generate_image(input, output):
    input_image = input.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

    #output_image = input_image + output_image

    # Clip values to [0, 1] if necessary
    input_image = np.clip(input_image, 0, 1)
    output_image = np.clip(output_image, 0, 1)

    print(input_image.shape)
    print(output_image.shape)

    # Display images
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(input_image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(output_image)
    axs[1].set_title('Output Image')
    axs[1].axis('off')

def tensor_to_cv(tensor_image):
    cv_image = cv.cvtColor(tensor_image.numpy().transpose(1, 2, 0), cv.COLOR_BGR2RGB)
    return cv_image

# THis function should return a pre-trained model and the augmentation necesary
def get_model():
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    model = prepare_model(model_name="UNet")
    model_path = "Outputs/overfitting/UNet_resize_Mixed_L1/model_checkpoints/UNet_model_epoch_149.pth"
    model.load_state_dict(torch.load(model_path, weights_only=False))
    # Disable dropout and batzh normilization
    model.eval()
    
    transform = prepare_preprocessor('resize', 512)

    return model, transform


# This function should apply the model with the given input and return the output
def apply_model(model, transform, input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Right now it doesn't do anything, since we dont have a model
    input = transform(input)

    input = input/255

    input = input.unsqueeze(0)

    input = input.to(device)
    model.to(device)

    output = model(input)
    #prediction = model(processed_input)

    return output

def process_image(image, model, transform):
    input_tensor = transform(image)
    input_tensor = input_tensor / 255
    output_tensor = apply_model(model, transform, image)
    generate_image(input_tensor, output_tensor)


def main():
    img = Image.open("Data/LOL-v2/Real_captured/Train/Normal/normal00002.png")
    #img.show()

    model, preprocess = get_model()
    process_image(img, model, preprocess)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

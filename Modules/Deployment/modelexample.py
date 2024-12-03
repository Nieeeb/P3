from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from torch import nn
import torch
import requests

def tensor_to_cv(tensor_image):
    cv_image = cv.cvtColor(tensor_image.numpy().transpose(1, 2, 0), cv.COLOR_BGR2RGB)
    return cv_image

# THis function should return a pre-trained model and the preprocessing necesary
def get_model():
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    # Disable dropout and batzh normilization
    model.eval()

    return model, preprocess


# This function should apply the model with the given input and return the output
def apply_model(model, preprocess, input):
    # Right now it doesn't do anything, since we dont have a model
    output = input
    #prediction = model(processed_input)

    return output


def main():
    img = Image.open("Data/LOL-v2/Real_captured/Train/Normal/normal00002.png")
    img.show()

    model, preprocess = get_model()
    processed = apply_model(model, preprocess, img)
    processed.show()


if __name__ == "__main__":
    main()

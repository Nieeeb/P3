# File containing the various processing methods used

# Paper 1: https://ieeexplore.ieee.org/document/10208804 (FF-Former: Swin Fourier Transformer for Nighttime Flare Removal)

# What they've done:
# "We crop the input flare free and flare-corrupted images into 512×512 with batch size of 2 to train our FF-Former.
# We use the Adam with β1 = 0.9 and β2 = 0.99 to optimize the Light Source Mask Loss Function.
# We only use Charbonnier L1 loss function [13] and set α to 0.05.
# The initial learning rate is 1e-4 and we use CosineAnnealingLR with 600,000 maximum iterations and 1e-7 minimum learning rate to adjust learning rate. 
# We also use horizontal and vertical flip for data enhancement."

from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import torch
import cv2 as cv
import math

class FixedHeightResize:
    def __init__(self, size) -> None:
        self.size = size
    
    def __call__(self, img):
        w = img.shape[2]
        h = img.shape[3]
        aspect_ratio = float(w) / float(h)
        new_w = math.ceil(self.size / aspect_ratio)
        return F.resize(img, (self.size, new_w), antialias=True)
        
     

def tensor_to_cv(tensor_image):
    cv_image = cv.cvtColor(tensor_image.numpy().transpose(1, 2, 0), cv.COLOR_BGR2RGB)
    return cv_image

def preprocessing_pipeline_example():
    transforms = T.Compose([
        # Convert to tensor
        T.ToImage(),
        # Resize image to a given size
        T.Resize(256, antialias=True),
        # Croping
        T.CenterCrop(242), # Crop at center
        #T.RandomCrop(242), # Crop randomly
        #T.RandomResizedCrop(242), # Crop and rezise at the same time
        # Flipping
        T.RandomVerticalFlip(0.5), # 50% of the time the image is flipped
        T.RandomHorizontalFlip(0.5),
        # Rotation
        T.RandomRotation((0, 180)), # Randomly rotate between 0 and 180 degrees
        # Normalization
        #T.ToDtype(torch.float32, scale=True), # Normalize expects float dtype
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms

# https://arxiv.org/abs/2011.12485
# "How to Train Neural Networks for Flare Removal, google"
def cropping_only_pipeline(size):
    transform = T.Compose([
        T.ToImage(),
        # Croping
        T.CenterCrop(size) # Crop
    ])
    return transform

# https://arxiv.org/pdf/2407.12431v1
# "GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval"
def resize_pipeline(size):
    transform = T.Compose([
        T.ToImage(),
        #FixedHeightResize(size),
        T.Resize((size, size)),
    ])
    return transform

# https://ieeexplore.ieee.org/document/10208804
# "FF-Former: Swin Fourier Transformer for Nighttime Flare Removal"
def crop_flip_pipeline(size):
    transform = T.Compose([
        T.ToImage(),
        # Croping
        T.CenterCrop(size), # Crop
        # Flipping
        T.RandomVerticalFlip(0.5), # 50% of the time the image is flipped
        T.RandomHorizontalFlip(0.5)
    ])
    return transform

# https://ieeexplore.ieee.org/document/10177254
# "Glow in the Dark: Low-Light Image Enhancement With External Memory"
def random_crop_and_flip_pipeline(size):
        transform = T.Compose([
        T.ToImage(),
        # Croping
        T.RandomCrop(size), # Crop
        # Flipping
        T.RandomVerticalFlip(0.5), # 50% of the time the image is flipped
        T.RandomHorizontalFlip(0.5)
    ])


def main():
    image = Image.open("Data/LOL-v2/Synthetic/Train/Low/r028896d3t.png")
    image.show()
    transforms = preprocessing_pipeline_example()
    image = transforms(image)
    image = F.to_pil_image(image)
    image.show()
    pass

if __name__ == "__main__":
    main()
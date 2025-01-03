# Stuff we need for deployment:
# Trained model. Probably a checkpoint?
    # How the model is structued. We need to set up the skeleton and then apply the trained weights
# Dataloader for data. Needs to take an image input and convert it into the format the model expects
# Going from dataloader to model, and then catching the output of the model
# Formatting the output and turning it into a format the user can see. Back into an image
# Stuff for the user
    # Way to connect to the model. webpage?
    # Way to upload images
    # Way to see output
    # Before and after?
    # Performance metrics before and after? PSNR etc.

import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from helpers import process_image, get_model
import matplotlib.pyplot as plt
import requests
from PIL import Image
import timeit

# Create app object 
app = FastAPI()

model, augmentation = get_model()

# API Endpoints
@app.get('/')
def index():
    return {'Hello': 'Welcome to low light model, access the api docs and test the API at http://0.0.0.0:8000/docs#/.'}

@app.post("/enhance")
async def upload(file: UploadFile = File(...)):
    try:
        tic = timeit.default_timer()
        path = f"image.png"
        contents = file.file.read()
        with open(path, 'wb') as f:
            f.write(contents)
        input = Image.open(path)
        process_image(input, model, augmentation, tic)
        save_path = f"image.png"
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=e)
    finally:
        file.file.close()

    return FileResponse(save_path)

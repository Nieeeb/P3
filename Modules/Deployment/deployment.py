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

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from Modules.Deployment.modelexample import get_model, apply_model
from typing import Annotated

# Create app object 
app = FastAPI()

model, preprocess = get_model()



# API Endpoints
@app.get('/')
def index():
    return {'Hello': 'Welcome to low light model, access the api docs and test the API at http://0.0.0.0:8000/docs#/.'}

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post('/enhance')
def perform_enhancement():
    image = create_file()
    output = apply_model(image)
    return FileResponse(output)

def main():
    client = TestClient(app)
    respone = client.get("/")
    print(respone)
    response = client.get("/enhance")
    print(response)

if __name__ == "__main__":
    main()

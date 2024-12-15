import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to sys.path to ensure modules can be found
root_dir = os.getcwd()
sys.path.append(root_dir)

# Import the U_Net model class
#from Models.model_zoo.U_Net import U_Net
#from Models.model_zoo.CAN import CANModel
# Import the test dataset loader and any required preprocessing pipelines
from Models.src.CLARITY_dataloader import LolTestDatasetLoader, LolDatasetLoader, LolValidationDatasetLoader
from Modules.Preprocessing.preprocessing import crop_flip_pipeline, resize_pipeline
from Modules.checkpointing import prepare_model, load_latest_checkpoint, prepare_preprocessor
from Models.model_zoo.seqentialmodel import SequentialModel

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_model_dicts():
    datasets = ['Mixed', 'LowLight', 'LensFlare']

    augmentations = ['resize']

    model_names = ['UNet']

    modeldicts = []

    loss = ['charbonnier', 'charbonnier_weighted']


    # model_Mixed_Charbonnier = prepare_model('UNet')
    # model_Mixed_Charbonnier.load_state_dict(torch.load(r"Outputs\Models\UNet_resize_Mixed_charbonnier\model_checkpoints\UNet_model_epoch_49.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_UnWeightedLoss_mixed"
    # model_dict = {'model': model_Mixed_Charbonnier, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)  


    # model_Mixed_Charbonnier_Weighted = prepare_model('UNet')
    # model_Mixed_Charbonnier_Weighted.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_Mixed_charbonnier_weighted\model_checkpoints\UNet_model_epoch_49Weighted.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_WeightedLoss_mixed"
    # model_dict = {'model': model_Mixed_Charbonnier_Weighted, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)



    model_Mixed_Charbonnier_Weighted10 = prepare_model('UNet')
    model_Mixed_Charbonnier_Weighted10.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_Mixed_charbonnier_weighted\model_checkpoints\UNet_model_epoch_49Weighted10.pth", map_location='cpu', weights_only=False))
    model_name = f"Sequence_WeightedLoss_mixed10"
    model_dict = {'model': model_Mixed_Charbonnier_Weighted10, 'modelname': model_name, 'transform': 'resize'}
    modeldicts.append(model_dict)

    # model_Mixed_Charbonnier_Weighted15 = prepare_model('UNet')
    # model_Mixed_Charbonnier_Weighted15.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_Mixed_charbonnier_weighted\model_checkpoints\UNet_model_epoch_49Weighted15.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_WeightedLoss_mixed15"
    # model_dict = {'model': model_Mixed_Charbonnier_Weighted15, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)



    # model_Low = prepare_model('UNet')
    # model_Low.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LowLight_charbonnier\model_checkpoints\UNet_model_epoch_49.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_LL_mixed"
    # model_dict = {'model': model_Low, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)



    # model_Lens = prepare_model('UNet')
    # model_Lens.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LensFlare_charbonnier\model_checkpoints\UNet_model_epoch_49LFNotDark.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_LF"
    # model_dict = {'model': model_Lens, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)

    # model_Lensdark = prepare_model('UNet')
    # model_Lensdark.load_state_dict(torch.load(r"C:\Users\Victor Steinrud\Documents\DAKI\3. semester\P3\Outputs\Models\UNet_resize_LensFlare_charbonnier\model_checkpoints\UNet_model_epoch_49LFDARK.pth", map_location='cpu', weights_only=False))
    # model_name = f"Sequence_LFDARK_mixed"
    # model_dict = {'model': model_Lensdark, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)


    # sequential1 = SequentialModel(model_Lens, model_Low)
    # model_name = f"Sequence_LF_LL_mixed"
    # model_dict = {'model': sequential1, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)
 
    # sequential2 = SequentialModel(model_Low, model_Lens)
    # model_name = f"Sequence_LL_LF_mixed"
    # model_dict = {'model': sequential2, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)
    


    # sequential3 = SequentialModel(model_Lensdark, model_Low)
    # model_name = f"Sequence_LFDARK_LL_mixed"
    # model_dict = {'model': sequential3, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)
 
    # sequential4 = SequentialModel(model_Low, model_Lensdark)
    # model_name = f"Sequence_LL_LFDARK_mixed"
    # model_dict = {'model': sequential4, 'modelname': model_name, 'transform': 'resize'}
    # modeldicts.append(model_dict)
    return modeldicts

def generate_image(modeldict, input, target):
    # Testing loop
    with torch.no_grad():
        model = modeldict['model']
        model.eval()
        # Move data to the appropriate device
        input = input.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(input)
        #print(outputs)

        # Convert tensors to numpy arrays for matplotlib
        # Remove the batch dimension and permute dimensions to (H, W, C)
        input_image = input.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = outputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
        target_image = target.cpu().squeeze(0).permute(1, 2, 0).numpy()

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

        #plt.show()

        plt.savefig(f"Outputs/ValidationImages/{modeldict['modelname']}.jpg")
        #plt.imsave(f"Outputs/ValidationImages/{modeldict['modelname']}.jpg", axs)

modeldicts = generate_model_dicts()

tranform = prepare_preprocessor('resize', 512)
dataset = LolValidationDatasetLoader(flare=True, transform=tranform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
input, target = next(iter(dataloader))

for modeldict in modeldicts:
    generate_image(modeldict, input, target)


print("Images written")

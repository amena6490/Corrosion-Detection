import os 
from show_results__ import*
from tqdm import tqdm   
import torch

# Load the trained model, you could possibly change the device from cpu to gpu if 
# you have your gpu configured.
model = torch.load(f'./stored_weights/l1_loss/weights_100.pt', map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

source_image_dir = './source_dir/'
destination_mask = './predicted_masks_l1_loss_corrosion_progression/'
destination_overlays = './combined_overlays_l1_loss_corrosion_progression/'

from PIL import Image
import os
import glob
import numpy as np

for image_name in tqdm(os.listdir(source_image_dir)):
    print(image_name)
    image_path = source_image_dir + image_name
    generate_images(model, image_path, image_name, destination_mask, destination_overlays)

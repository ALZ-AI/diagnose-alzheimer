"""
import torch
import torchvision
from torchvision import datasets, transforms
from utils.settings import *
import os
import utils.yaml

params = utils.yaml.read_yaml("params.yaml")



labels = {
    0: "MildDemented",
    1: "ModerateDemented",
    2: "NonDemented",
    3: "VeryMildDemented"
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        params["mean"],
        params["std"]
        )
    ])


#os.rmdir(ALZHEIMER_PROCESSED_DATA_DIR)


train_dataset = datasets.ImageFolder(TRAIN_RAW_DATA_DIR, transform=transform)
train_set = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)"""

import os
from utils.settings import *
import cv2
import glob
import shutil

folders = {
    "train": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
    "test": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
}

def process(path, path_to_save):
    filename = os.path.basename(path)
    pass


shutil.rmtree(ALZHEIMER_PROCESSED_DATA_DIR)


for key, labels in folders.items():
    for label in labels:
        # Create Directories for images to be processed
        saved_to_path = os.path.join(ALZHEIMER_PROCESSED_DATA_DIR, key, label)
        os.makedirs(saved_to_path, exist_ok=True)
        
        for image_path in glob.glob(os.path.join(ALZHEIMER_RAW_DATA_DIR, key, label, "*.jpg")):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (88, 104), interpolation = cv2.INTER_NEAREST)
            # Save Processed Image
            filename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(saved_to_path, filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
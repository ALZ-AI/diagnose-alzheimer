import os
from utils.settings import *
import cv2
import glob
import shutil

folders = {
    "train": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
    "test": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
}

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
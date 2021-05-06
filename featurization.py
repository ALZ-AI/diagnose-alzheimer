from utils.settings import *
import torch
from torchvision import transforms, datasets
import utils.yaml

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((44, 52)),
    transforms.ToTensor()
    ])



train_dataset = datasets.ImageFolder(TRAIN_RAW_DATA_DIR, transform=transform)
train_set = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_image_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_image_count += image_count_in_a_batch

    mean /= total_image_count
    std /= total_image_count
    return mean.tolist(), std.tolist()

mean, std = get_mean_and_std(train_set)

data = utils.yaml.read_yaml("normalization.yaml")
data["dataset"]["mean"] = mean
data["dataset"]["std"] = std
utils.yaml.save_yaml("normalization.yaml", data)
import pandas as pd
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Project directory path
base_dir = 'S:\\Mv_py_ass'

# Define paths
counting_dir = os.path.join(base_dir, 'counting')


class AppleDataset(Dataset):
    def __init__(self, data_category, transform=None):
        images_path = os.path.join(counting_dir, data_category, 'images')
        annotations_path = os.path.join(counting_dir, data_category, f'{data_category}_ground_truth.txt')
        # Check if your TXT file has a header row; if it does, set header=0, otherwise header=None
        self.annotations = pd.read_csv(annotations_path, header=0)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]  # No need to cast to int, it should already be the correct type
        # Ensure that the label is within the expected range [0, 6]
        label = torch.eye(7)[min(max(label, 0), 6)]

        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.Grayscale(num_output_channels=1),  # Convert to single channel grayscale
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
])


# Function to process and save datasets
def process_and_save_dataset(data_category, file_name):
    dataset = AppleDataset(data_category=data_category, transform=transform)
    images = []
    labels = []

    for i in range(len(dataset)):
        img, lbl = dataset[i]
        images.append(img)
        labels.append(lbl)

        if i % 100 == 0:  # Print progress every 100 images
            print(f"Processed {i} images")

    images = torch.stack(images)
    labels = torch.stack(labels)

    torch.save({'images': images, 'labels': labels}, file_name)
    print(f"Saved dataset to {file_name}")


# Process and save datasets
process_and_save_dataset('train', os.path.join(base_dir, 'train_dataset.pth'))
process_and_save_dataset('val', os.path.join(base_dir, 'val_dataset.pth'))
process_and_save_dataset('test', os.path.join(base_dir, 'test_dataset.pth'))

import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from joint_unet import UNetWithClassification


class UnlabeledLesionDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.images_dir = os.path.join(dataset_path)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for category, label in zip(["benign", "malignant"], [0, 1]):
            img_folder = os.path.join(self.images_dir, category)
            img_files = sorted(os.listdir(img_folder))

            for img_file in img_files:
                self.image_paths.append(os.path.join(img_folder, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        classification_label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, classification_label


class HospitalLesionDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        self.images_dir = os.path.join(dataset_path, "images")
        self.masks_dir = os.path.join(dataset_path, "masks")
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        for category, label in zip(["benign", "malignant"], [0, 1]):
            img_folder = os.path.join(self.images_dir, category)
            mask_folder = os.path.join(self.masks_dir, category)

            img_files = sorted(os.listdir(img_folder))
            mask_files = sorted(os.listdir(mask_folder))

            for img_file, mask_file in zip(img_files, mask_files):
                self.image_paths.append(os.path.join(img_folder, img_file))
                self.mask_paths.append(os.path.join(mask_folder, mask_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        classification_label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, mask, classification_label


class UltrasoundDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.transforms = transforms

        self.benign_dir = os.path.join(self.dataset_path, "benign")
        self.malignant_dir = os.path.join(self.dataset_path, "malignant")

        if not os.path.isdir(self.benign_dir) or not os.path.isdir(self.malignant_dir):
            raise FileNotFoundError(
                "The dataset directory structure is incorrect. Expected 'benign' and 'malignant' subdirectories.")

        self.benign_images = [os.path.join(self.benign_dir, f) for f in os.listdir(self.benign_dir) if
                              f.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.tif'))]
        self.malignant_images = [os.path.join(self.malignant_dir, f) for f in os.listdir(self.malignant_dir) if
                                 f.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.tif'))]

        self.images = self.benign_images + self.malignant_images
        self.labels = [0] * len(self.benign_images) + [1] * len(self.malignant_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label = self.labels[index]

        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    target_transform = transforms.ToTensor()
    dataset = HospitalLesionDataset("lesion_segmentation", transform=transform, target_transform=target_transform,
                                    clinical_features_csv="lumc_rdgg_attributes.csv")
    # image, mask_tensor, classification_label = dataset[1]
    # image = image.unsqueeze(0)
    # joint_unet = UNetWithClassification(3, 2, 2)
    # prediction = joint_unet(image)
    # print()

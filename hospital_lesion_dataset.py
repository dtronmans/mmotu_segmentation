import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from joint_unet import UNetWithClassification


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


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    target_transform = transforms.ToTensor()
    dataset = HospitalLesionDataset("lesion_segmentation", transform=transform, target_transform=target_transform)
    image, mask_tensor, classification_label = dataset[1]
    image = image.unsqueeze(0)
    joint_unet = UNetWithClassification(3, 2, 2)
    prediction = joint_unet(image)
    print()

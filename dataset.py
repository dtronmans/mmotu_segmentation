import os
import torchvision.transforms.functional as TF
import random
from threading import Thread

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MMOTUSegmentationDataset(Dataset):

    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "images")
        self.annotated_dir = os.path.join(dataset_path, "annotations")
        self.transforms = transforms

        self.filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.filenames)

    def display(self, im):
        im.show()

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        file_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.annotated_dir, file_name).replace(".JPG", "_binary.PNG")

        original_image = Image.open(file_path).convert("L")
        mask_image = Image.open(mask_path).convert("L")

        if self.transforms:
            original_image = self.transforms(original_image)
            mask_image = self.transforms(mask_image)

        if random.random() > 0.5:
            original_image = TF.hflip(original_image)
            mask_image = TF.hflip(mask_image)

        if random.random() > 0.5:
            original_image = TF.vflip(original_image)
            mask_image = TF.vflip(mask_image)

        return {"original": original_image, "mask": mask_image}


if __name__ == "__main__":
    dataset = MMOTUSegmentationDataset("otu_2d/OTU_2d")
    for i in range(0, 100):
        dataset_item = dataset[i]
    print()

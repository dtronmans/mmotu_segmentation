import os
import torchvision.transforms.functional as TF
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledDataset(Dataset):

    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.transforms = transforms

        self.filenames = sorted(os.listdir(self.dataset_path))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        file_path = os.path.join(self.dataset_path, file_name)

        image = Image.open(file_path).convert("L")
        image = self.transforms(image)
        return image


class MMOTUSegmentationDataset(Dataset):

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "images")
        self.annotated_dir = os.path.join(dataset_path, "annotations")
        self.transform = transform

        self.filenames = sorted(os.listdir(self.image_dir))
        self.resize = transforms.Resize((384, 384))

    def __len__(self):
        return len(self.filenames)

    def display(self, im):
        im.show()

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        file_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.annotated_dir, file_name).replace(".JPG", "_binary.PNG")

        original_image = Image.open(file_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        original_image = self.resize(original_image)
        mask_image = self.resize(mask_image)

        mask_image = TF.to_tensor(mask_image)
        mask_image = (mask_image > 0).float()

        if random.random() > 0.5:
            original_image = TF.hflip(original_image)
            mask_image = TF.hflip(mask_image)

        if random.random() > 0.5:
            original_image = TF.vflip(original_image)
            mask_image = TF.vflip(mask_image)

        # Apply image-only augmentations
        if self.transform:
            original_image = self.transform(original_image)

        return {"original": original_image, "mask": mask_image}


if __name__ == "__main__":
    dataset = MMOTUSegmentationDataset("otu_2d/OTU_2d")
    for i in range(0, 100):
        dataset_item = dataset[i]
    print()

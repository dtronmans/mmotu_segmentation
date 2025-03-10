import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MultimodalHospitalLesionDataset(Dataset):

    def __init__(self, dataset_path, csv_path, transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform

        self.data = pd.read_csv(csv_path)

        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.clinical_features = []

        for _, row in self.data.iterrows():
            study_id = row['Study ID']
            malignancy = row['Malignancy status']
            menopausal_status = 1 if row['Menopausal status'] == 2 else row['Menopausal status']
            hospital = 0 if study_id.startswith("RDG") else 1

            category = "benign" if malignancy == 0 else "malignant"
            img_folder = os.path.join(self.dataset_path, "images", category)
            mask_folder = os.path.join(self.dataset_path, "masks", category)

            # Ensure the folder exists before listing files
            if os.path.exists(img_folder) and os.path.exists(mask_folder):
                image_files = sorted([f for f in os.listdir(img_folder) if f.startswith(study_id)])
                mask_files = sorted([f for f in os.listdir(mask_folder) if f.startswith(study_id)])

                for img_file, mask_file in zip(image_files, mask_files):
                    self.image_paths.append(os.path.join(img_folder, img_file))
                    self.mask_paths.append(os.path.join(mask_folder, mask_file))
                    self.labels.append(malignancy)
                    self.clinical_features.append([menopausal_status, hospital])

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
        clinical_features = self.clinical_features[index]

        return image, mask, classification_label, clinical_features


if __name__ == "__main__":
    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()
    dataset = MultimodalHospitalLesionDataset("lesion_segmentation", "lesion_segmentation/patient_attributes.csv", transform, target_transform)
    item = dataset[0]

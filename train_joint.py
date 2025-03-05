import torch
from torchvision import transforms

from hospital_lesion_dataset import HospitalLesionDataset
from joint_unet import UNetWithClassification

if __name__ == "__main__":
    model = UNetWithClassification(3, 2, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()

    dataset = HospitalLesionDataset("lesion_segmentation", transform=transform, target_transform=target_transform)



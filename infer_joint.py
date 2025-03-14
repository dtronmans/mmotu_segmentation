import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from hospital_lesion_dataset import HospitalLesionDataset, UnlabeledLesionDataset
from joint_unet import transfer_unet_weights, UNetWithClassification
from multimodal_hospital_lesion_dataset import MultimodalHospitalLesionDataset


def infer_directory(model, dataloader, unlabeled=True):
    for batch in dataloader:
        clinical_information = None
        if unlabeled:
            image, classification_labels = batch
        else:
            image, masks, classification_labels, clinical_information = batch
        predicted = model(image, clinical_information)

        image_prediction = torch.sigmoid(predicted[0].squeeze())
        image_prediction = (image_prediction >= 0.5).float()

        class_prediction = int(torch.sigmoid(predicted[1]) > 0.5)
        # Convert image and mask to NumPy
        numpy_mask = image_prediction.detach().cpu().numpy()
        numpy_image = image.squeeze().permute(1, 2,
                                              0).detach().cpu().numpy()  # Convert from Tensor (C, H, W) to (H, W, C)

        # Normalize image for visualization
        numpy_image = (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())  # Normalize to [0,1]
        numpy_image = (numpy_image * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

        # Convert mask to heatmap (red color)
        mask_colored = np.zeros_like(numpy_image)
        mask_colored[:, :, 2] = (numpy_mask * 255).astype(np.uint8)  # Apply red color to mask

        # Blend the mask with the original image
        overlay = cv2.addWeighted(numpy_image, 0.7, mask_colored, 0.3, 0)

        # Show overlay
        cv2.imshow("Segmentation Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("True class: Benign" if classification_labels == 0 else "True class: Malignant")
        print("Predicted class: Benign" if class_prediction == 0 else "Predicted class: Malignant")


if __name__ == "__main__":
    model = UNetWithClassification(3, 1, 1, use_clinical_features=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("multimodal_joint_unet_intermediate.pt", weights_only=True, map_location=device))
    model.eval()
    #
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((336, 544))])
    target_transform = transforms.ToTensor()
    #
    # dataset = UnlabeledLesionDataset("lesion_segmentation", transform)
    dataset = MultimodalHospitalLesionDataset("lesion_segmentation", "lesion_segmentation/patient_attributes.csv",
                                              transform, target_transform)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    #
    infer_directory(model, dataloader, unlabeled=False)

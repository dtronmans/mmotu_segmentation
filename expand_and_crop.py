import os

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from architecture import UNet
from infer import infer_on_single_image


def process_image_for_classification(image, model, transform, expansion_factor=1.2):
    # Step 1: Get the binary mask from the segmentation model
    _, binary_mask = infer_on_single_image(image, model, transform, histogram=False)

    # Ensure the mask is binary (in case it's not exactly 0/255)
    binary_mask = (binary_mask > 127).astype(np.uint8) * 255

    # Step 2: Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contour found!")
        return None, None

    # Step 3: Find the largest contour (assuming it corresponds to the lesion)
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 4: Fit an ellipse around the contour
    if len(largest_contour) < 5:
        print("Not enough points to fit an ellipse!")
        return None, None

    ellipse = cv2.fitEllipse(largest_contour)
    (x, y), (major_axis, minor_axis), angle = ellipse

    # Expand the ellipse
    major_axis *= expansion_factor
    minor_axis *= expansion_factor

    # Step 5: Create a mask with the expanded ellipse
    ellipse_mask = np.zeros_like(binary_mask)
    cv2.ellipse(ellipse_mask, ((int(x), int(y)), (int(major_axis), int(minor_axis)), angle), 255, -1)

    opened_image = Image.open(image).convert("L")  # Ensure grayscale
    image_tensor = transform(opened_image).unsqueeze(0)  # Apply transform
    image_np = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)  # Convert to uint8

    # Step 6: Apply the mask to the original image
    processed_image = cv2.bitwise_and(image_np, image_np, mask=ellipse_mask)

    if processed_image is not None:
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_image


if __name__ == "__main__":
    model = UNet(1, 1)
    model.load_state_dict(torch.load("mmotu_segm_new.pt", weights_only=True, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])

    images_path = os.path.join("benign", "images")
    for image in os.listdir(images_path):
        cropped_image = process_image_for_classification(os.path.join(images_path, image), model, transform)

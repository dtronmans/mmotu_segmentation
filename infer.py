import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.spatial.distance import pdist, squareform
from torchvision import transforms

from architecture import UNet


def infer_on_single_image(image_path, model, transform):
    image = Image.open(image_path).convert("L")
    ground_truth_path = image_path.replace("images", "annotations")
    ground_truth_path = ground_truth_path.replace(".jpg", "_binary.png")
    ground_truth_path = ground_truth_path.replace(".JPG", "_binary.PNG")
    ground_truth = Image.open(ground_truth_path).convert("L")

    image_tensor = transform(image).unsqueeze(0)
    ground_truth_tensor = transform(ground_truth)

    with torch.no_grad():
        prediction = model(image_tensor)
        predicted_mask = torch.sigmoid(prediction).squeeze().numpy()

    binary_mask = (predicted_mask > 0.5).astype("uint8") * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour[:, 0, :]

        dist_matrix = squareform(pdist(points))
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

        point1 = tuple(points[max_dist_idx[0]])
        point2 = tuple(points[max_dist_idx[1]])
        max_diameter = np.linalg.norm(np.array(point1) - np.array(point2))

        # Draw the maximum diameter line
        binary_mask_with_line = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(binary_mask_with_line, point1, point2, (0, 255, 0), 2)
        cv2.putText(
            binary_mask_with_line, f"{int(max_diameter)} px",
            (point1[0], point1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Display original image and mask
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(image_tensor.squeeze(), cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(binary_mask_with_line)
    ax[1].set_title(f"Lesion Max Diameter: {int(max_diameter)} px")
    ax[1].axis("off")

    ax[2].imshow(ground_truth_tensor.squeeze(), cmap="gray")
    ax[2].set_title("Ground Truth Mask")
    ax[2].axis("off")

    plt.show()

    return max_diameter

if __name__ == "__main__":
    model = UNet(1, 1)
    model.load_state_dict(torch.load("mmotu_segmentation.pt", weights_only=True, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((288, 288)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to tensor
    ])

    images_path = os.path.join("otu_2d", "OTU_2d", "images")
    for image in os.listdir(images_path):
        infer_on_single_image(os.path.join(images_path, image), model, transform)


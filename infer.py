import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader
from torchvision import transforms

from architecture import UNet
from dataset import MMOTUSegmentationDataset, UnlabeledDataset
from expand_and_crop import crop_and_expand_image


def adapt_batch_norm(model, dataloader, device):
    model.train()

    with torch.no_grad():
        for batch in dataloader:
            images = batch.to(device)
            _ = model(images)  # Forward pass

    model.eval()  # Switch back to eval mode
    return model


def load_histogram_data(histogram_path="average_histogram.npz"):
    data = np.load(histogram_path)
    return data["histogram"], data["cdf"]


def apply_histogram_matching(image, ref_cdf):
    img_hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    img_cdf = np.cumsum(img_hist)
    img_cdf = img_cdf / img_cdf[-1]

    matched_values = np.interp(img_cdf, ref_cdf, np.arange(256))

    matched_image = np.interp(image.flatten(), bin_edges[:-1], matched_values)
    matched_image = matched_image.reshape(image.shape).astype(np.uint8)

    return matched_image


def infer(image_path, model, transform, histogram_path="average_histogram.npz", histogram=False,
          ground_truth_exists=False, expand_and_crop=False):
    image = Image.open(image_path).convert("L")

    if histogram:
        avg_hist, ref_cdf = load_histogram_data(histogram_path)
        image_np = np.array(image)
        matched_image_np = apply_histogram_matching(image_np, ref_cdf)
        image = Image.fromarray(matched_image_np)

    image_tensor = transform(image).unsqueeze(0)

    if ground_truth_exists:
        ground_truth_path = image_path.replace("images", "annotations")
        ground_truth_path = ground_truth_path.replace(".jpg", ".png")
        ground_truth_path = ground_truth_path.replace(".JPG", ".PNG")
        ground_truth = Image.open(ground_truth_path).convert("L")
        ground_truth_tensor = transform(ground_truth)
    else:
        ground_truth_tensor = None

    with torch.no_grad():
        prediction = model(image_tensor)
        predicted_mask = torch.sigmoid(prediction).squeeze().numpy()

    binary_mask = (predicted_mask > 0.5).astype("uint8") * 255
    black_pixel_ratio = calculate_black_pixel_ratio(image_tensor, binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_mask_with_line, max_diameter = draw_largest_contour(contours, binary_mask=binary_mask)

    image_np = image_tensor.squeeze().squeeze().numpy() * 255
    overlay = cv2.addWeighted(image_np, 0.5, binary_mask, 0.5, 0, dtype=cv2.CV_32F)

    cropped_image = None
    if expand_and_crop:
        cropped_image = crop_and_expand_image(image_path, binary_mask, transform)

    display_results(image_tensor, binary_mask_with_line, overlay, max_diameter, black_pixel_ratio, ground_truth_exists,
                    ground_truth_tensor, cropped_image)

    return max_diameter, binary_mask


def display_results(image_tensor, binary_mask_with_line, overlay, max_diameter, black_pixel_ratio,
                    ground_truth_exists=False,
                    ground_truth_tensor=None,
                    cropped_image=None):
    display_columns = 4 if ground_truth_exists else 3
    display_columns = display_columns + 1 if cropped_image is not None else display_columns

    fig, ax = plt.subplots(1, display_columns, figsize=(12, 5))
    ax[0].imshow(image_tensor.squeeze(), cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(binary_mask_with_line)
    ax[1].set_title(f"Lesion Max Diameter: {int(max_diameter)} px")
    ax[1].axis("off")

    ax[2].imshow(overlay, cmap="gray")
    ax[2].set_title(f"Mask Overlay")
    ax[2].axis("off")

    if ground_truth_exists and ground_truth_tensor:
        ax[3].imshow(ground_truth_tensor.squeeze(), cmap="gray")
        ax[3].set_title("Ground Truth Mask")
        ax[3].axis("off")

    if cropped_image is not None:
        ax[-1].imshow(cropped_image, cmap="gray")
        ax[-1].set_title(f"Cropped image\nBlack Pixel Ratio: {black_pixel_ratio:.2f}")
        ax[-1].axis("off")

    plt.show()


def draw_largest_contour(contours, binary_mask):
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

        return binary_mask_with_line, max_diameter
    else:
        raise RuntimeError("no contours found int the semantic mask")


def calculate_black_pixel_ratio(image_tensor, binary_mask):
    image_np = image_tensor.squeeze().numpy() * 255
    masked_pixels = image_np[binary_mask == 255]

    masked_pixels = min_max_normalize_255(masked_pixels)

    if masked_pixels.size == 0:
        return 0.0

    print(np.min(masked_pixels))
    black_pixels = np.sum(masked_pixels < 80)
    ratio = black_pixels / masked_pixels.size
    return ratio


def min_max_normalize_255(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_max == arr_min:
        return np.zeros_like(arr)  # Avoid division by zero if all values are the same

    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr * 255

if __name__ == "__main__":
    model = UNet(1, 1)
    model.load_state_dict(torch.load("mmotu_50.pt", weights_only=True, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    our_dataset = UnlabeledDataset("benign/images", transforms=transform)
    our_dataloader = DataLoader(our_dataset, batch_size=4, shuffle=False)
    # model = adapt_batch_norm(model, our_dataloader, "cpu")

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    images_path = os.path.join("LUMC_RDG_only_ovary", "malignant")
    for image in os.listdir(images_path):
        max_diameter, binary_mask = infer(os.path.join(images_path, image), model, transform, histogram=True,
                                          expand_and_crop=True)
        print()

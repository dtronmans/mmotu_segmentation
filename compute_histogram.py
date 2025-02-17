import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image




def compute_average_histogram(dataset_path, save_path="average_histogram.npz"):
    all_histograms = []
    num_bins = 256

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]

    if len(image_files) == 0:
        print("No images found in the dataset directory!")
        return

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        if img is None:
            print(f"Skipping {img_file}, could not read image.")
            continue

        # Compute histogram
        hist, _ = np.histogram(img, bins=num_bins, range=(0, 256))
        all_histograms.append(hist)

    avg_histogram = np.mean(all_histograms, axis=0)

    cdf = np.cumsum(avg_histogram)
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]

    np.savez(save_path, histogram=avg_histogram, cdf=cdf)
    print(f"Saved average histogram and CDF to {save_path}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_histogram, color='black')
    plt.title("Average Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.plot(cdf, color='blue')
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Cumulative Sum")

    plt.show()


if __name__ == "__main__":
    dataset_path = "otu_2d/OTU_2d/images"  # Change this to your dataset path
    compute_average_histogram(dataset_path, save_path="average_histogram.npz")

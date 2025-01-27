import os

from torchvision.transforms import transforms
from tqdm import tqdm

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from architecture import UNet
from dataset import MMOTUSegmentationDataset

if __name__ == "__main__":
    model = UNet(1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((288, 512)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to tensor
    ])

    dataset = MMOTUSegmentationDataset(os.path.join("otu_2d", "OTU_2d"), transforms=transform)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    losses = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            images = batch['original'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            predicted = model(images)
            loss = losses(predicted, masks)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Train Loss: {loss.item()}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images = batch['original'].to(device)
                masks = batch['mask'].to(device)

                predicted = model(images)
                loss = losses(predicted, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

    torch.save(model.state_dict(), "mmotu_segmentation.pt")

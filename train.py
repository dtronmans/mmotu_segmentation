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
    model = UNet(3, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MMOTUSegmentationDataset(os.path.join("/exports", "lkeb-hpc", "dzrogmans", "OTU_2d"), transform=transform)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    losses = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    num_epochs = 4000

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            images = batch['original'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            predicted = model(images)
            masks = (masks > 0).float()
            loss = losses(predicted, masks)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.2f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images = batch['original'].to(device)
                masks = batch['mask'].to(device)

                predicted = model(images)
                masks = (masks > 0).float()
                loss = losses(predicted, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.2f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "mmotu_intermediate.pt")

    torch.save(model.state_dict(), "mmotu_segmentation.pt")

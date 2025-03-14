import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from hospital_lesion_dataset import HospitalLesionDataset
from joint_unet import UNetWithClassification, transfer_unet_weights

if __name__ == "__main__":
    model = transfer_unet_weights("mmotu_intermediate_2.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()

    dataset = HospitalLesionDataset("lesion_segmentation", transform=transform, target_transform=target_transform)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    losses = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    num_epochs = 300

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            images, masks, classification_labels = batch
            images, masks, classification_labels = images.to(device), masks.to(device), classification_labels.to(device)

            optimizer.zero_grad()
            predicted = model(images)
            masks = (masks > 0).float()
            segmentation_loss = BCEWithLogitsLoss()(predicted[0], masks)
            classification_loss = BCEWithLogitsLoss()(predicted[1].squeeze(), classification_labels.float())
            total_loss = segmentation_loss + 0.5 * classification_loss
            train_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, masks, classification_labels = batch
                images, masks, classification_labels = images.to(device), masks.to(device), classification_labels.to(
                    device)

                predicted = model(images)
                masks = (masks > 0).float()
                segmentation_loss = BCEWithLogitsLoss()(predicted[0], masks)
                classification_loss = BCEWithLogitsLoss()(predicted[1].squeeze(), classification_labels.float())
                total_loss = segmentation_loss + 0.5 * classification_loss
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), "joint_unet_intermediate.pth")

    torch.save(model.state_dict(), "joint_unet_final.pt")
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from hospital_lesion_dataset import UltrasoundDataset
from joint_unet import UNetWithClassification, only_classification_model

if __name__ == "__main__":
    model = UNetWithClassification(3, 1, 1)
    model = only_classification_model(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("joint_unet_normalized.pt", weights_only=True, map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((336, 544))
    ])

    dataset = UltrasoundDataset("rdgg_sorted", transforms=transform)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    losses = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
    num_epochs = 300

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            images, classification_labels = batch
            images, classification_labels = images.to(device), classification_labels.to(device)

            optimizer.zero_grad()
            _, class_logits = model(images)
            classification_loss = BCEWithLogitsLoss()(class_logits.squeeze(), classification_labels.float())
            train_loss += classification_loss.item()

            classification_loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, classification_labels = batch
                images, classification_labels = images.to(device), classification_labels.to(
                    device)

                _, classification_logits = model(images)
                classification_loss = BCEWithLogitsLoss()(classification_logits.squeeze(), classification_labels.float())
                val_loss += classification_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    for param in model.up1.parameters():
        param.requires_grad = True
    for param in model.up2.parameters():
        param.requires_grad = True
    for param in model.up3.parameters():
        param.requires_grad = True
    for param in model.up4.parameters():
        param.requires_grad = True
    for param in model.outc.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), "joint_unet_after_classification.pt")

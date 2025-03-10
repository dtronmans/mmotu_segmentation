import torch
import torch.nn as nn

from architecture import DoubleConv, Down, Up, OutConv


class UNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes,
                 use_clinical_features=False, num_clinical_features=2, bilinear=False):
        super(UNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear
        self.use_clinical_features = use_clinical_features

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_segmentation_classes))

        self.feature_dim = 1024 // factor

        self.classifier = nn.Linear(self.feature_dim, num_classification_classes)

        # Clinical Features Encoding (Maps clinical features to the same space as x5)
        if self.use_clinical_features:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(num_clinical_features, 16),  # Map clinical features to 16D
                nn.ReLU(),
                nn.Linear(16, self.feature_dim),  # Project to same latent space
                nn.Sigmoid()  # Produces scaling factors (like attention)
            )

    def forward(self, x, clinical_features=None):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        seg_logits = self.outc(x_seg)

        class_features = x5.view(x5.shape[0], -1)

        if self.use_clinical_features and clinical_features is not None:
            clinical_embedding = self.clinical_encoder(clinical_features)  # (batch, feature_dim)
            class_features = class_features * (1 + clinical_embedding)  # FiLM: Modulate instead of adding

        class_logits = self.classifier(class_features)  # Classification output

        return seg_logits, class_logits


def only_classification_model(model):
    model.train()

    for param in model.up1.parameters():
        param.requires_grad = False
    for param in model.up2.parameters():
        param.requires_grad = False
    for param in model.up3.parameters():
        param.requires_grad = False
    for param in model.up4.parameters():
        param.requires_grad = False
    for param in model.outc.parameters():
        param.requires_grad = False


def transfer_unet_weights(unet_model_path):
    joint_model = UNetWithClassification(3, 1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_weights = torch.load(unet_model_path, weights_only=True, map_location=device)

    joint_model_state = joint_model.state_dict()

    transfer_weights = {k: v for k, v in unet_weights.items() if
                        k in joint_model_state and v.shape == joint_model_state[k].shape}

    joint_model_state.update(transfer_weights)
    joint_model.load_state_dict(joint_model_state)

    print(f"Transferred {len(transfer_weights)}/{len(unet_weights)} layers from U-Net to joint model.")

    return joint_model

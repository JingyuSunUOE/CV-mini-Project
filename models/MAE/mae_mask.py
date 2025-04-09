import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegDataset
import models.MAE.models_mae as models_mae

class MAEMaskDecoder(nn.Module):
    def __init__(self, mae_model_path=None, num_classes=3, num_mask_tokens=8, hidden_dim=256):
        super().__init__()

        if mae_model_path == "logs/MAE_Base_original/best_model.pt":
            mae_model = torch.load(mae_model_path,weights_only=False)

            # If the model is saved as DataParallel, remove the DataParallel module
            if isinstance(mae_model, torch.nn.DataParallel):
                mae_model = mae_model.module

            self.image_encoder = mae_model

        else:
            # self.image_encoder = models_mae.__dict__['mae_vit_large_patch16']()
            self.image_encoder = models_mae.__dict__['mae_vit_base_patch16']()

            if mae_model_path:
                checkpoint = torch.load(mae_model_path)
                checkpoint_model = checkpoint['model']
                self.image_encoder.load_state_dict(checkpoint_model, strict=False)

        # Freeze the image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # self.proj_dim = 1024  # ViT-Large Output Dimensions
        self.proj_dim = 768  # ViT-Base Output Dimensions

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Parameter(torch.randn(1, num_mask_tokens, hidden_dim))

        self.input_proj = nn.Linear(self.proj_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.class_proj = nn.Linear(hidden_dim, num_classes)

        self.position_embedding = nn.Parameter(torch.randn(1, 196, hidden_dim))

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_classes, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def encode_patch_tokens(self, x):
        latent, _, _ = self.image_encoder.forward_encoder(x, mask_ratio=0.0)
        latent = latent[:, 1:, :]  # 🚨Discard the first [CLS] token!
        return latent  # → shape = [B, 196, 1024]

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, 224, 224, 3] → [B, 3, 224, 224]

        with torch.no_grad():
            x_encoded = self.encode_patch_tokens(x)  # → [B, N=196, C=1024]

        B, N, C = x_encoded.shape
        x_proj = self.input_proj(x_encoded)  # → [B, N, hidden_dim]
        x_proj = x_proj + self.position_embedding

        H_feat = W_feat = int(N ** 0.5)
        feat_map = x_proj.permute(0, 2, 1).reshape(B, -1, H_feat, W_feat)

        mask_tokens = self.mask_tokens.expand(B, -1, -1)  # (B, K, D)
        image_tokens = x_proj.permute(1, 0, 2)  # (N, B, D)
        mask_queries = mask_tokens.permute(1, 0, 2)  # (K, B, D)

        decoded = self.transformer_decoder(mask_queries, image_tokens)  # (K, B, D)
        decoded = decoded.permute(1, 0, 2)  # (B, K, D)

        masks = torch.einsum('bkd,bdhw->bkhw', decoded, feat_map)
        masks = masks.permute(0, 1, 2, 3)

        class_logits = self.class_proj(decoded)  # (B, K, num_classes)
        class_logits = class_logits.permute(0, 2, 1)  # (B, num_classes, K)

        segmentation = torch.einsum('bck,bkxy->bcxy', class_logits.softmax(dim=2), masks)
        segmentation = self.decoder(segmentation)

        return segmentation

# Test the MAEMaskDecoder model
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # mae_path = "../../checkpoints/MAE/mae_pretrain_vit_large.pth"
#     mae_path = "../../checkpoints/MAE/mae_pretrain_vit_base.pth"
#     model = MAEMaskDecoder(mae_model_path=mae_path, num_classes=3, num_mask_tokens=8).to(device)
#     print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
#
#     # Set the loss function, because the mask is not binary, dog is 1, cat is 2, background is 0, so use CrossEntropyLoss
#     criterion = nn.CrossEntropyLoss()
#
#     # Test forward
#     train_dataset = SegDataset(root_dir="../../new_dataset", split="train", transform=True) # Apply augmentations
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
#     for images, masks, text_description in train_loader:
#         images, masks = images.to(device), masks.to(device)
#         print("Input shape:", images.shape)
#         print("Mask shape:", masks.shape)
#
#         output = model(images)
#         print("Output shape:", output.shape)
#
#         # Convert the one-hot mask to a class index mask
#         masks = masks.argmax(dim=1)
#
#         # Calculate the loss
#         loss = criterion(output, masks)
#         print("Loss:", loss.item())
#
#         break

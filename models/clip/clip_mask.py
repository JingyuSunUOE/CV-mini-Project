import torch
import torch.nn as nn


class CLIPMaskDecoder(nn.Module):
    def __init__(self, clip_model, num_classes=3, num_mask_tokens=8, hidden_dim=256):
        super().__init__()
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual

        self.input_resolution = self.image_encoder.input_resolution  # 224
        self.patch_size = 14              # 14
        self.grid_size = self.input_resolution // self.patch_size    # 16
        self.proj_dim = 1024                # 1024

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Parameter(torch.randn(1, num_mask_tokens, hidden_dim))

        self.input_proj = nn.Linear(self.proj_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        # mask projection
        self.class_proj = nn.Linear(hidden_dim, num_classes)

        self.position_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim))

        self.decoder = nn.Sequential(
            # 16 → 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_classes, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 32 → 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 64 → 112
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, kernel_size=1)  # logits: [B, 3, 224, 224]
        )

    def encode_patch_tokens(self, x):
        # Step 1: conv1 cuts the image into patches
        x = self.image_encoder.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid*grid)
        x = x.permute(0, 2, 1)  # (B, num_patches, width)

        # Step 2: Add positional embedding (skip the 0th position CLS)
        pos_embed = self.image_encoder.positional_embedding  # shape: [197, 1024]
        pos_embed = pos_embed[1:]  # → [196, 1024]
        x = x + pos_embed.unsqueeze(0)  # (1, 196, 1024)

        # Step 3: Transformer Pre-Processing
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [num_patches, B, width]
        x = self.image_encoder.transformer(x)  # ViT encoding
        x = x.permute(1, 0, 2)  # [B, num_patches, width]
        x = self.image_encoder.ln_post(x)
        return x  # patch tokens: [B, 196, 1024]

    def forward(self, x):
        # x: (B, 224, 224, 3) → permute to (B, 3, 224, 224)
        x = x.permute(0, 3, 1, 2)

        # Step 1: encode with CLIP ViT encoder (output shape: [B, num_patches+1, C])
        with torch.no_grad():
            x_encoded = self.encode_patch_tokens(x)  # instead of self.image_encoder(x)

        B, N, C = x_encoded.shape
        x_proj = self.input_proj(x_encoded)  # → [B, N, hidden_dim]
        x_proj = x_proj + self.position_embedding

        # Reshape to 2D feature map: (B, hidden_dim, H, W)
        H_feat = W_feat = int(N ** 0.5)
        feat_map = x_proj.permute(0, 2, 1).reshape(B, -1, H_feat, W_feat)

        # Step 2: mask tokens
        mask_tokens = self.mask_tokens.expand(B, -1, -1)  # (B, K, D)

        # TransformerDecoder expects (T, B, D)
        image_tokens = x_proj.permute(1, 0, 2)      # (N, B, D)
        mask_queries = mask_tokens.permute(1, 0, 2) # (K, B, D)

        decoded = self.transformer_decoder(mask_queries, image_tokens)  # (K, B, D)
        decoded = decoded.permute(1, 0, 2)  # (B, K, D)

        # Step 3: get mask maps from decoded tokens and image feature map
        masks = torch.einsum('bkd,bdhw->bkhw', decoded, feat_map)  # (B, K, H', W')
        masks = masks.permute(0, 1, 2, 3)  # (B, K, H', W')

        # Step 4: classify each mask token
        class_logits = self.class_proj(decoded)  # (B, K, num_classes)
        class_logits = class_logits.permute(0, 2, 1)  # (B, num_classes, K)

        # Step 5: fuse masks with class logits
        segmentation = torch.einsum('bck,bkxy->bcxy', class_logits.softmax(dim=2), masks)  # (B, num_classes, H', W')

        # Step 6: upsample to 224x224
        segmentation = self.decoder(segmentation)

        return segmentation  # (B, 3, 224, 224)

# Test the CLIPMaskDecoder model
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     clip_model, preprocess = clip.load("ViT-L/14", device=device)
#     clip_model.float()
#     # Freeze the clip model
#     for param in clip_model.parameters():
#         param.requires_grad = False
#     model = CLIPMaskDecoder(clip_model=clip_model, num_classes=3, num_mask_tokens=8).to(device)
#     print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
#
#     # Set the loss function, because the mask is not binary, dog is 1, cat is 2, background is 0, so use CrossEntropyLoss
#     criterion = nn.CrossEntropyLoss()
#
#     # Test forward
#     train_dataset = SegDataset(root_dir="../new_dataset", split="train", transform=True) # Apply augmentations
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

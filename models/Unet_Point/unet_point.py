import torch
import torch.nn as nn
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from dataset import SegDataset
import numpy as np
import matplotlib.pyplot as plt


def generate_click_and_heatmap_binary(onehot_mask, sigma=1.0):
    """
    Input:
    onehot_mask: numpy array, shape (H, W, C), one-hot encoded mask
    sigma: standard deviation of Gaussian kernel, controls heatmap attenuation
    Output:
    click_point: tuple (x, y)
    heatmap: numpy array, shape (H, W)
    updated_binary_mask: numpy array, shape (H, W), binary map
    """
    H, W, C = onehot_mask.shape

    # Step 1: Convert one-hot mask to binary mask (foreground = 1, background = 0)
    # Assume background is channel 0
    binary_mask = np.any(onehot_mask[:, :, 1:], axis=-1).astype(np.uint8)  # shape: (H, W)

    # Step 2: Click a random pixel
    y, x = np.random.randint(0, H), np.random.randint(0, W)
    click_point = (x, y)

    # Step 3: Determine whether the click is in the foreground or background
    if binary_mask[y, x] == 1:
        # Click in the foreground, keep it as it is
        updated_binary_mask = binary_mask.copy()
    else:
        # Click on the background to reverse: foreground becomes background, background becomes foreground
        updated_binary_mask = 1 - binary_mask

    # Step 4: Generate heatmap
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist_sq = (yy - y) ** 2 + (xx - x) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / np.max(heatmap)

    return click_point, heatmap, updated_binary_mask


class UNetPointSeg(nn.Module):
    def __init__(self):
        super().__init__()
        # Use UNet2DModel as the backbone, with configurations adjusted according to task requirements:
        # - Input size: 224×224, input channels 3, output channels set to the last block_out_channels (256)
        # - Use a shallower structure: block_out_channels=(64, 128, 256) and layers_per_block=2
        # - Both downsample and upsample layers use ResNet sampling method, combined with self-attention modules to capture global information
        # - dropout set to 0.1, helps alleviate overfitting issues with small datasets
        self.backbone = UNet2DModel(
            sample_size=224,  # Input/output image size is 224×224
            in_channels=4,  # Input is RGB image
            out_channels=1,  # backbone output channels, equal to the last item in block_out_channels
            center_input_sample=True,  # Input image is centered (can be enabled based on data preprocessing)
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(64, 128, 256),  # The output channels of each block
            layers_per_block=2,  # Each block contains 2 convolutional layers
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type="resnet",  # Use ResNet style downsampling
            upsample_type="resnet",  # Use ResNet style upsampling
            dropout=0.1,  # Dropout rate
            act_fn="silu",
            attention_head_dim=4,  # Attention head dimension
            norm_num_groups=16,  # Number of groups for normalization
            attn_norm_num_groups=None,
            norm_eps=1e-5,
            resnet_time_scale_shift="default",
            class_embed_type=None,  # No class embedding
            num_class_embeds=None
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (B, 224, 224, 4)
        Returns:
            seg_mask: Segmentation mask, shape (B, 4, 224, 224)
        """
        # Transpose the input from (B, 224, 224, 4) to (B, 4, 224, 224)
        if x.dim() == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)

        B = x.size(0)
        # For segmentation tasks, the temporal information is not critical, so use dummy timestep (all zeros)
        dummy_timesteps = torch.zeros(B, dtype=torch.long, device=x.device)

        # Get the backbone output, which is a UNet2DOutput object, extract the actual Tensor features
        backbone_output = self.backbone(x, dummy_timesteps)
        # Get the actual Tensor features
        seg_mask = backbone_output.sample

        return seg_mask


# Test the UNetPointSeg model
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNetPointSeg().to(device)
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
#         print("Text Description:", text_description)
#
#         heatmaps = []
#         updated_masks = []
#         for i in range(images.shape[0]):
#             click_point, heatmap, updated_binary_mask = generate_click_and_heatmap_binary(masks[i].permute(1,2,0).cpu().detach().numpy())
#             heatmaps.append(heatmap)
#             updated_masks.append(updated_binary_mask)
#
#             print("Click point:", click_point)
#             print("Heatmap shape:", heatmap.shape)
#             print("Updated mask shape:", updated_binary_mask.shape)
#             print("Mask shape:", masks[i].shape)
#
#             mask_combine = masks[i].argmax(dim=0)
#             plt.imshow(images[i].cpu().detach().numpy())
#             plt.title("Image")
#             plt.colorbar()
#             plt.show()
#
#             plt.imshow(mask_combine.cpu().detach().numpy())
#             plt.title("Mask")
#             plt.colorbar()
#             plt.show()
#
#             plt.imshow(updated_binary_mask)
#             plt.title("Updated Mask")
#             plt.colorbar()
#             plt.show()
#
#             plt.imshow(heatmap)
#             plt.title("Heatmap")
#             plt.colorbar()
#             plt.show()
#             break
#
#         heatmaps = np.array(heatmaps)
#         heatmaps = torch.from_numpy(heatmaps).float()  # <--- 转为 float32 (B, 224, 224)
#         # Convert to (B, 224, 224, 1)
#         heatmaps = heatmaps.unsqueeze(-1)
#         heatmaps = heatmaps.to(device)
#         print("Heatmaps shape:", heatmaps.shape)
#
#         updated_masks = np.array(updated_masks)
#         updated_masks = torch.from_numpy(updated_masks)
#         updated_masks = updated_masks.to(device) # (B, 224, 224)
#         # Convert to (B, 1, 224, 224)
#         updated_masks = updated_masks.unsqueeze(1)
#         print("Updated masks shape:", updated_masks.shape)
#
        # break

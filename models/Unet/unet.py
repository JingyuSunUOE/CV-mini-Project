import torch
import torch.nn as nn
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from dataset import SegDataset


class UNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        # Use UNet2DModel as the backbone, with configurations adjusted according to task requirements:
        # - Input size: 224×224, input channels 3, output channels set to the last block_out_channels (256)
        # - Use a shallower structure: block_out_channels=(64, 128, 256) and layers_per_block=2
        # - Both downsample and upsample layers use ResNet sampling method, combined with self-attention modules to capture global information
        # - dropout set to 0.1, helps alleviate overfitting issues with small datasets
        self.backbone = UNet2DModel(
            sample_size=224,  # Input/output image size is 224×224
            in_channels=3,  # Input is RGB image
            out_channels=256,  # backbone output channels, equal to the last item in block_out_channels
            center_input_sample=True,  # Input image is centered (can be enabled based on data preprocessing)
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
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

        # Segmentation head: Use a 1×1 convolution to map the 256 channels of the backbone output to 1 output channel
        # Then get the segmentation mask, background is 0, dog is 1, cat is 2
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (B, 224, 224, 3)
        Returns:
            seg_mask: Segmentation mask, shape (B, 3, 224, 224)
        """
        # Transpose the input from (B, 224, 224, 3) to (B, 3, 224, 224)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        B = x.size(0)
        # For segmentation tasks, the temporal information is not critical, so use dummy timestep (all zeros)
        dummy_timesteps = torch.zeros(B, dtype=torch.long, device=x.device)

        # Get the backbone output, which is a UNet2DOutput object, extract the actual Tensor features
        backbone_output = self.backbone(x, dummy_timesteps)
        # Get the actual Tensor features
        features = backbone_output.sample

        # Use a 1×1 convolution to map the 256 channels to 3 output channel, the output shape is (B, 3, 224, 224)
        seg_mask = self.seg_head(features)

        return seg_mask


# Test the UNetSeg model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSeg().to(device)
    print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set the loss function, because the mask is not binary, dog is 1, cat is 2, background is 0, so use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Test forward
    train_dataset = SegDataset(root_dir="../../new_dataset", split="train", transform=True) # Apply augmentations
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    for images, masks, text_description in train_loader:
        images, masks = images.to(device), masks.to(device)
        print("Input shape:", images.shape)
        print("Mask shape:", masks.shape)

        # # 用于存放中间层特征
        # features = {}
        #
        # # 定义 hook 函数
        # def hook_fn(module, input, output):
        #     features["mid_block"] = output.detach()
        #
        # # 注册 hook 到 mid_block
        # hook_handle = model.backbone.mid_block.register_forward_hook(hook_fn)
        #
        # # 前向传播
        # with torch.no_grad():
        #     _ = model(images)
        #
        # # 提取结果
        # mid_feat = features["mid_block"]  # shape: (1, 256, 56, 56)
        # print("Mid-block feature shape:", mid_feat.shape)
        #
        # # 记得在不需要后解绑 hook（可选）
        # hook_handle.remove()

        output = model(images)
        print("Output shape:", output.shape)

        # Convert the one-hot mask to a class index mask
        masks = masks.argmax(dim=1)

        # Calculate the loss
        loss = criterion(output, masks)
        print("Loss:", loss.item())

        break

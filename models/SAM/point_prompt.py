import torch
import torch.nn as nn
from typing import Tuple
from dataset import SegDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1  # [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # (B, N, C)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate dense positional encoding of image feature map
        Args:
        size: (H, W) spatial size of image feature map
        Returns:
        pos_encoding: (1, C, H, W)
        """
        H, W = size
        device = self.positional_encoding_gaussian_matrix.device

        y_embed = torch.linspace(0, H - 1, H, device=device)
        x_embed = torch.linspace(0, W - 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_embed, x_embed, indexing="ij")
        coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        coords = coords.reshape(1, H * W, 2)  # (1, HW, 2)

        pe = self._pe_encoding(coords)  # (1, HW, C)
        pe = pe.view(1, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (1, C, H, W)
        return pe


class PointPromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, input_image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 点的类型 embedding：正点、负点、padding
        self.pos_point_embed = nn.Embedding(1, embed_dim)       # label = 1
        self.neg_point_embed = nn.Embedding(1, embed_dim)       # label = 0
        self.not_a_point_embed = nn.Embedding(1, embed_dim)     # label = -1

    def forward(self, points: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Encode point prompts.

        points: Tuple of (coords_tensor, labels_tensor)
            - coords_tensor: (B, N, 2), pixel coordinates
            - labels_tensor: (B, N), values in {1, 0, -1}
        Returns:
            sparse_embeddings: (B, N, embed_dim)
        """
        coords, labels = points
        coords = coords + 0.5  # shift to center of pixel
        pe = self.pe_layer.forward_with_coords(coords, self.input_image_size)  # (B, N, embed_dim)

        # Add corresponding embedding according to label
        sparse_embedding = torch.zeros_like(pe)

        sparse_embedding[labels == -1] += self.not_a_point_embed.weight
        sparse_embedding[labels == 0] += self.neg_point_embed.weight
        sparse_embedding[labels == 1] += self.pos_point_embed.weight

        pe[labels == -1] = 0.0  # Avoid position encoding interfering with padding

        return pe + sparse_embedding

def generate_point_prompt_and_binary_mask(masks_onehot: torch.Tensor):
    """
    对每个样本，随机从非背景类别中选一个前景类，采样一个正点，返回该点和该类的 binary 掩码。

    参数：
        masks_onehot: [B, C, H, W]，C为类别数（含背景）的one-hot掩码

    返回：
        coords_tensor: [B, 1, 2]，点坐标 (y, x)
        labels_tensor: [B, 1]，点标签，始终为正点 1
        binary_masks: [B, 1, H, W]，对应类的二值掩码（0=背景, 1=目标类区域）
        selected_classes: [B]，被采样的前景类别编号
    """
    B, C, H, W = masks_onehot.shape
    device = masks_onehot.device

    coords_list = []
    labels_list = []
    binary_masks_list = []
    selected_classes = []

    for i in range(B):
        mask = masks_onehot[i]  # [C, H, W]
        pixel_counts = mask.view(C, -1).sum(dim=1)
        valid_classes = (pixel_counts > 0).nonzero(as_tuple=False).view(-1)

        if len(valid_classes) == 0:
            selected_class = torch.tensor(1, device=device)
        else:
            selected_class = valid_classes[torch.randint(0, len(valid_classes), (1,)).item()]

        selected_classes.append(selected_class)

        # 获取该类的掩码，随机从其像素中采样一点
        class_mask = mask[selected_class]  # [H, W]
        ys, xs = torch.where(class_mask > 0)
        if len(xs) == 0:
            y, x = torch.tensor(0), torch.tensor(0)
        else:
            idx = torch.randint(0, len(xs), (1,))
            y, x = ys[idx], xs[idx]

        # 得到该样本的语义标签图
        semantic_label_map = torch.argmax(mask, dim=0)  # [H, W]
        clicked_class = semantic_label_map[y, x].item()

        coords_list.append(torch.tensor([[y.item(), x.item()]], device=device))  # [1, 2]
        labels_list.append(torch.tensor([1], device=device))  # 正点

        # 如果点击的是背景 → 反转掩码作为 binary
        if clicked_class == 0:
            binary_mask = (semantic_label_map == 0).float()  # 仅背景位置为1，其它为0
        else:
            binary_mask = (semantic_label_map == clicked_class).float()

        binary_masks_list.append(binary_mask.unsqueeze(0))  # [1, H, W]

    coords_tensor = torch.stack(coords_list, dim=0)      # [B, 1, 2]
    labels_tensor = torch.stack(labels_list, dim=0)      # [B, 1]
    binary_masks = torch.stack(binary_masks_list, dim=0) # [B, 1, H, W]
    selected_classes = torch.stack(selected_classes, dim=0)  # [B]

    return coords_tensor, labels_tensor, binary_masks, selected_classes


# Test the encoder
# if __name__ == "__main__":
#
#     encoder = PointPromptEncoder(embed_dim=256, input_image_size=(224, 224))
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder.to(device)
#
#     # Create dataset instances
#     train_dataset = SegDataset(root_dir="../../new_dataset", split="train", transform=True) # Apply augmentations
#     print("The length of the dataset is: ", len(train_dataset))
#
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
#
#     # 仅检查一个 batch
#     for images, masks, text_description in train_loader:
#         print("Batch Image shape:", images.shape)  # (B, H, W, 3)
#         print("Batch Mask shape:", masks.shape)  # (B, 3, H, W)
#         print("Text Description:", text_description)
#
#         images, masks = images.to(device), masks.to(device)
#
#         # 通过 generate_point_prompt_and_binary_mask 得到随机采样的点、标签以及更新后的 mask
#         coords_tensor, labels_tensor, binary_masks, selected_classes = generate_point_prompt_and_binary_mask(masks)
#
#         print("coords_tensor shape:", coords_tensor.shape)  # (B, 1, 2)
#         print("labels_tensor shape:", labels_tensor.shape)  # (B, 1)
#         print("binary_masks shape:", binary_masks.shape)  # (B, 1, H, W)
#
#         sparse_embeddings = encoder((coords_tensor.to(device), labels_tensor.to(device))) # ([B, 1, 256])
#         print("Sparse embeddings shape:", sparse_embeddings.shape)
#
#         image_pe = encoder.pe_layer((14, 14)) # ([1, 256,14,14])
#         print("Image PE shape:", image_pe.shape)
#
#         # 将图像和 mask 转到 CPU 并转换为 numpy
#         images_np = images.cpu().numpy()  # [B, H, W, 3]
#         masks_np = masks.cpu().numpy()  # [B, 3, H, W]
#         binary_masks_np = binary_masks.cpu().numpy()  # [B, 1, H, W]
#         coords_np = coords_tensor.cpu().numpy()  # [B, 1, 2]
#         selected_classes_np = selected_classes.cpu().numpy()  # [B]
#
#         # 可视化每个样本
#         for i in range(images_np.shape[0]):
#             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#             img = images_np[i]
#             mask_label = np.argmax(masks_np[i], axis=0)
#             binary_mask = binary_masks_np[i, 0]
#             y, x = coords_np[i, 0]
#
#             # 原图
#             axes[0].imshow(img)
#             axes[0].scatter(x, y, c='red', s=40, label='Prompt Point')
#             axes[0].set_title("Input Image")
#             axes[0].legend()
#
#             # 原始 one-hot 合成的 mask
#             im1 = axes[1].imshow(mask_label, cmap='jet')
#             axes[1].scatter(x, y, c='red', s=40)
#             axes[1].set_title(f"Original Mask (multi-class)\nSelected Class: {selected_classes_np[i]}")
#             plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
#
#             # binary mask（选中类）
#             im2 = axes[2].imshow(binary_mask, cmap='gray')
#             axes[2].scatter(x, y, c='red', s=40)
#             axes[2].set_title("Binary Mask (selected class)")
#             plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
#
#             plt.tight_layout()
#             plt.show()
#
#         break  # 只检查第一个 batch

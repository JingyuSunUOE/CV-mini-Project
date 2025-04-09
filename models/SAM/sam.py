import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SAM.point_prompt import PointPromptEncoder
from models.SAM.sam_encoder import ImageEncoderViT
from models.SAM.sam_decoder import MaskDecoder
from models.SAM.transformer import TwoWayTransformer
from models.SAM.point_prompt import generate_point_prompt_and_binary_mask
from dataset import SegDataset
from torch.utils.data import DataLoader

class SAM(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalized image: expects image as [B, 3, H, W], range [0, 255] or [0, 1]
        """
        image_device = image.device

        if image.max() > 1.0:
            image = image / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406],device=image_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_device).view(1, 3, 1, 1)
        return (image - mean) / std

    def get_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def get_prompt_embeddings(self, coords_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> torch.Tensor:
        return self.prompt_encoder((coords_tensor, labels_tensor))

    def get_output(self, image_embeddings, image_pe, sparse_embeddings, multimask_output):
        return self.mask_decoder(image_embeddings, image_pe, sparse_embeddings, multimask_output)

    def forward(
        self,
        image: torch.Tensor,             # [B, H, W, 3]
        coords_tensor: torch.Tensor, # (B, N, 2), pixel coordinates
        labels_tensor: torch.Tensor # (B, N)
        ):

        image = image.permute(0, 3, 1, 2) # [B, 3. H, W]
        image = self.preprocess(image)

        image_embeddings = self.get_image_embeddings(image)  # ([B, 256, 14, 14])

        sparse_embeddings = self.get_prompt_embeddings(coords_tensor, labels_tensor)  # ([B, 1, 256])

        image_pe = self.prompt_encoder.pe_layer((14, 14))  # ([1, 256,14,14])

        output = self.get_output(image_embeddings,image_pe,sparse_embeddings,multimask_output=False) # ([B, 1, 56, 56])
        output = output[0]

        predicted_masks = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
        predicted_masks = predicted_masks[..., :224, :224]

        return predicted_masks

# Test the SAM
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     decoder = MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=256,
#                 mlp_dim=2048,
#                 num_heads=8,))
#     print("The number of parameters in the decoder: ", sum(p.numel() for p in decoder.parameters()))
#     point_encoder = PointPromptEncoder(embed_dim=256, input_image_size=(224, 224))
#     print("The number of parameters in the point_encoder: ", sum(p.numel() for p in point_encoder.parameters()))
#     image_encoder = ImageEncoderViT(img_size=224, depth=4, num_heads=4, global_attn_indexes=[2, 5, 8, 11])
#     print("The number of parameters in the image_encoder: ", sum(p.numel() for p in image_encoder.parameters()))
#     model = SAM(image_encoder, point_encoder, decoder).to(device)
#     print("The number of parameters in the models: ", sum(p.numel() for p in model.parameters()))
#
#     # Create dataset instances
#     train_dataset = SegDataset(root_dir="../../new_dataset", split="train", transform=True) # Apply augmentations
#     print("The length of the dataset is: ", len(train_dataset))
#
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
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
#         coords_tensor = coords_tensor.to(device)
#         labels_tensor = labels_tensor.to(device)
#         binary_masks = binary_masks.to(device)
#
#         print("coords_tensor shape:", coords_tensor.shape)  # (B, 1, 2)
#         print("labels_tensor shape:", labels_tensor.shape)  # (B, 1)
#         print("updated_masks_onehot shape:", binary_masks.shape)  # (B, 1, H, W)
#
#         predicted_masks = model(images, coords_tensor, labels_tensor) # (B, 1, H, W)
#         print("Predicted Masks shape:", predicted_masks.shape)
#
#         # Calculate the loss
#         loss = F.binary_cross_entropy_with_logits(predicted_masks, binary_masks)
#         print("Loss:", loss.item())
#
#         break

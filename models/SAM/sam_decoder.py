import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Type

from models.SAM.common import LayerNorm2d
from models.SAM.transformer import TwoWayTransformer

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Select appropriate masks
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create output tokens (1 IOU + N mask tokens)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        # Combine prompt tokens with IOU + mask tokens
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand image features to match prompt token batch
        src = image_embeddings
        pos_src = image_pe

        b, c, h, w = src.shape

        # Run transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upsample image features
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Generate dynamic weights from mask tokens
        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)],
            dim=1,
        )

        # Apply dynamic heads to image features
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Predict IOU quality for each mask
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


# Test the decoder
# if __name__ == "__main__":
#     decoder = MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=256,
#                 mlp_dim=2048,
#                 num_heads=8,))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     decoder.to(device)
#
#     dummy_image_embeddings = torch.randn(2, 256, 14, 14).to(device)
#     dummy_image_pe = torch.randn(2, 256, 14, 14).to(device)
#     dummy_prompt_embeddings = torch.randn(2, 1, 256).to(device)
#
#     output = decoder(dummy_image_embeddings,dummy_image_pe,dummy_prompt_embeddings, multimask_output=False)
#     print("Output shape:", output[0].shape)
#
#     predicted_masks = F.interpolate(output[0], size=(224, 224), mode='bilinear', align_corners=False)
#     predicted_masks = predicted_masks[..., :224, :224]
#     print("Predicted masks shape:", predicted_masks.shape)

from typing import Tuple, Union
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

class UNETR2D_MAE(nn.Module):
    def __init__(
            self,
            mae_model,  # Pre-trained MAE model
            out_channels: int,
            img_size: Tuple[int, int],
            feature_size: int = 16,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
    ) -> None:
        super().__init__()
        # Use the pre-trained MAE model (only the encoder part)
        self.mae = mae_model

        # If self.mae.patch_embed.patch_size is already a tuple
        self.patch_size = self.mae.patch_embed.patch_size

        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )

        self.hidden_size = hidden_size

        # Keep the original input information (optional)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=3,  # Assume the input is an RGB image
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # The remaining encoder modules are used to upsample and fuse the intermediate features in the MAE
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=2,
            in_channels=feature_size,
            out_channels=out_channels
        )

    def proj_feat(self, x, hidden_size, feat_size):
        """
        Reconstruct the features extracted by the transformer (shape [B, num_patches, hidden_size]) into a 2D feature map [B, hidden_size, H, W].
        """
        x = x[:, 1:, :]  # Remove the cls token (if exists)
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in):
        # Transpose the input from (B, 224, 224, 3) to (B, 3, 224, 224)
        if x_in.dim() == 4 and x_in.shape[-1] == 3:
            x_in = x_in.permute(0, 3, 1, 2)

        # Note: Random masking is usually not needed in segmentation tasks, and mask_ratio can be set to 0
        latent, mask, ids_restore, hidden_states = self.mae.forward_encoder(x_in, mask_ratio=0)

        # Choose the appropriate intermediate layer as the skip connection (e.g. for MAE with depth=12, use layers 4, 7, 10)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states[2], self.hidden_size, self.feat_size))
        enc3 = self.encoder3(self.proj_feat(hidden_states[4], self.hidden_size, self.feat_size))
        enc4 = self.encoder4(self.proj_feat(hidden_states[6], self.hidden_size, self.feat_size))

        # Reshape the final latent representation into a 2D feature map
        dec4 = self.proj_feat(latent, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits

# Test the UNETR2D_MAE model
# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # mae_model = mae.__dict__['mae_vit_base_patch16']()
#     mae_model = mae.__dict__['mae_vit_tiny_patch16']()
#
#     mae_model.to(device)
#     print("The number of parameters:", sum(p.numel() for p in mae_model.parameters() if p.requires_grad))
#
#     # Freeze the MAE model
#     for param in mae_model.parameters():
#         param.requires_grad = False
#
#     model = UNETR2D_MAE(
#         mae_model=mae_model,
#         out_channels=3,
#         img_size=(224, 224),
#         feature_size=16,
#         hidden_size=384, # 1024 for mae_base
#         norm_name="instance",
#         conv_block=False,
#         res_block=True,
#     ).to(device)
#     print("The number of parameters:", sum(p.numel() for p in model.parameters()))
#     print("The trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
#
#     # Test forward
#     x = torch.randn(2, 224, 224, 3).to(device)
#     with torch.no_grad():
#         out = model(x)
#         print("Model output shape:", out.shape)

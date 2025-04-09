import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.clip.clip as clip

# --------------------------
# 1. Residual Convolution Unit
# --------------------------
class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not self.bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not self.bn)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


# --------------------------
# 2. FeatureFusionBlock
# --------------------------
class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, bn=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


# --------------------------
# 3. Decoder
# --------------------------
class Decoder(nn.Module):
    def __init__(self, features=512):
        super(Decoder, self).__init__()

        # Upsample progressively, ×2 each time
        self.fusion1 = FeatureFusionBlock_custom(features, activation=nn.ReLU(False), bn=True, align_corners=True)
        self.fusion2 = FeatureFusionBlock_custom(features, activation=nn.ReLU(False), bn=True, align_corners=True)
        self.fusion3 = FeatureFusionBlock_custom(features, activation=nn.ReLU(False), bn=True, align_corners=True)
        self.fusion4 = FeatureFusionBlock_custom(features, activation=nn.ReLU(False), bn=True, align_corners=True)

    def forward(self, x):
        x = self.fusion4(x)
        x = self.fusion3(x)
        x = self.fusion2(x)
        x = self.fusion1(x)
        return x


# --------------------------
# 4. Spatial Regularization Block（SRB）
# --------------------------
class SpatialRegBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialRegBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


# --------------------------
# 5. Segmentation Network LSegNet
# --------------------------
class LSegNet(nn.Module):
    def __init__(self, clip_model, text_features, num_classes=3, features=512):
        super(LSegNet, self).__init__()
        self.device = next(clip_model.parameters()).device

        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.num_classes = num_classes

        self.features = features

        # Training the Decoder
        self.decoder = Decoder(features=features)

        # Spatial Regularization Module (SRB)
        self.spatial_reg = SpatialRegBlock(in_channels=num_classes)

        # Calculate the scaling parameters of text features
        logit_init = torch.tensor(np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(logit_init)

        self.register_buffer("text_features", text_features)

        self.proj = None

        # Subtract 1 because the first token is [CLS]
        num_patches = self.image_encoder.positional_embedding.shape[0] - 1
        self.grid_size = int(num_patches ** 0.5)  # E.g., sqrt(256) = 16

        self.final_refine = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def extract_patch_features(self, x):
        """
        Extracts patch-level features from CLIP's visual transformer.
        Automatically infers grid size from positional embeddings.

        Args:
            x: Input image tensor of shape (B, 3, H, W)
        Returns:
            Tensor of shape (B, C, grid_size, grid_size)
        """
        dtype = self.image_encoder.conv1.weight.dtype
        x = x.to(dtype)

        # Step 1. Initial projection to patch embeddings
        x = self.image_encoder.conv1(x)  # -> (B, C, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # -> (B, num_patches, C)

        # Step 2. Add position embedding (excluding CLS token)
        pos_embed = self.image_encoder.positional_embedding[1:]  # drop CLS token
        x = x + pos_embed.unsqueeze(0).to(dtype)

        # Step 3. LayerNorm and Transformer
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # -> (num_patches, B, C)
        x = self.image_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # -> (B, num_patches, C)
        x = self.image_encoder.ln_post(x)

        # Step 4. Recover spatial shape (B, C, grid, grid)
        num_patches = x.shape[1]
        grid_size = int(num_patches ** 0.5)
        assert grid_size * grid_size == num_patches, "Non-square patch grid!"

        x = x.permute(0, 2, 1).contiguous().view(x.shape[0], -1, grid_size, grid_size)
        return x

    def forward(self, x):
        """
        Args:
            x: Input image, shape (B, 224, 224, 3)
        Returns:
            seg_logits: segmentation logits, shape (B, num_classes, H, W)
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW

        B, _, H, W = x.shape

        # 1. Get CLIP image features (B, C, 14, 14)
        img_features = self.extract_patch_features(x)
        # print("img_features shape:", img_features.shape)

        # Apply projection
        if self.proj is None:
            in_dim = img_features.shape[1]
            self.proj = nn.Sequential(
                nn.Conv2d(in_dim, self.features, kernel_size=1),
                nn.BatchNorm2d(self.features),
                nn.ReLU(inplace=True)
            ).to(img_features.dtype).to(img_features.device)

        img_features = self.proj(img_features)
        # print("img_features shape:", img_features.shape)

        img_features = F.normalize(img_features, p=2, dim=1)

        # 2. Send it to the Decoder
        img_features = img_features.to(self.decoder.fusion1.out_conv.weight.dtype)
        # print("img_features shape:", img_features.shape)
        img_features = self.decoder(img_features)  # Stepwise upsampling
        # print("img_features shape:", img_features.shape)
        img_features = F.interpolate(img_features, size=(224, 224), mode="bilinear", align_corners=True)
        # print("img_features shape:", img_features.shape)
        img_features = self.final_refine(img_features)
        # print("img_features shape:", img_features.shape)

        # 3. Calculate image and text similarity
        text_features = self.text_features.to(x.dtype)  # or x.device
        similarity = self.logit_scale * torch.einsum('bchw,kc->bkhw', img_features, text_features)
        # print("similarity shape:", similarity.shape)

        # 4. Adding a background channel
        background_logits = torch.zeros(B, 1, similarity.shape[2], similarity.shape[3], device=x.device)
        # print("background_logits shape:", background_logits.shape)
        logits = torch.cat([background_logits, similarity], dim=1)
        # print("logits shape:", logits.shape)

        # 5. Spatial Regularization
        seg_logits = self.spatial_reg(logits)

        return seg_logits

# --------------------------
# 6. Test
# --------------------------
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit_t_16 = {
        "embed_dim": 512,  # Output vector dimension: try 256 or 512
        "image_resolution": 224,  # Input image resolution
        "vision_layers": 6,  # Reduce to 8 layers (try between 6 and 8)
        "vision_width": 256,  # The hidden dimension is set to 512 (384 can also be considered)
        "vision_patch_size": 16,  # The size of each patch is 16x16
        "context_length": 77,  # Maximum number of tokens in text
        "vocab_size": 49408,  # Vocabulary size, usually unchanged
        "transformer_width": 256,  # Text Transformer width, consistent with vision_width
        "transformer_heads": 4,  # 512 // 64 = 8 heads (if changed to 384, it will be 6 heads)
        "transformer_layers": 4  # Reduced to 4-layer text Transformer
    }

    # clip_model = CLIP(**vit_t_16).to(device)

    # Load a pre-trained CLIP model (e.g. ViT-B/32)
    # 512
    # clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 768
    # clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # Define the category text, here only contains "dog" and "cat"
    # text_list = ["dog", "cat"]
    # text_features = get_text_features(text_list, clip_model, device)  # (2, C)

    # Freeze the clip model
    for param in clip_model.parameters():
        param.requires_grad = False

    print("The number of parameters in the clip models: ", sum(p.numel() for p in clip_model.parameters()))
    print("The number of trainable parameters in the clip models: ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad))

    # Define category prompts
    label_to_prompt = {
        "cat": "a photo of a cat",
        "dog": "a photo of a dog"
    }

    text_tokens = clip.tokenize(list(label_to_prompt.values())).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)

    print("text_features shape:", text_features.shape)

    # Build a segmentation model (the number of output channels is 3, corresponding to background, dog, cat)
    model = LSegNet(clip_model, text_features, num_classes=3, features=512).to(device)
    print("The number of parameters in the models: ", sum(p.numel() for p in model.parameters()))
    print("The number of trainable parameters in the models: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Construct a fake input: input image shape (B, 224, 224, 3), for example, batch size=2
    dummy_img = torch.randn(10, 224, 224, 3).to(device)

    # The model forward propagation obtains segmentation logits, the shape is expected to be (2, 3, 224, 224)
    seg_logits = model(dummy_img)
    print("seg_logits shape:", seg_logits.shape)

    # Assume your ground truth mask is one-hot encoded and has shape (B, 3, 224, 224)
    # Here we construct an example one-hot mask: randomly generate 0,1,2 labels and then convert them to one-hot
    gt_labels = torch.randint(0, 3, (10, 224, 224)).to(device)  # (B, H, W) Label format
    # If one-hot is needed, it can be converted, but the cross entropy loss can be directly used with the category index
    criterion = nn.CrossEntropyLoss()
    loss = criterion(seg_logits, gt_labels)
    print("loss:", loss.item())

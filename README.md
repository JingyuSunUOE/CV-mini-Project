# CV Mini Project: Point-prompted Segmentation

## Project Overview

In this project, we implemented and compared several deep learning models for **semantic image segmentation** using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).  
The objective was to accurately segment pet images into three classes: **background**, **cat**, and **dog**.

### Approaches Explored

We explored a variety of model architectures and training strategies:

- **Custom UNet**  
  A UNet model enhanced with **ResNet-style skip connections**, and attention-based **downsampling** and **upsampling** blocks to improve feature representation.
  
![Model Architecture](assets/unet.png)

- **MAE-Seg (Masked Autoencoder Segmentation)**  
which learns robust features by reconstructing randomly masked image regions. We froze the encoder used a query-based decoder inspired by recent works, introducing learnable mask tokens that interact with image tokens via a Transformer to extract semantic regions.

![Model Architecture](assets/auto2.png)
- **CLIP-Seg**  
  a cross-modal model with both image and text encoders. The approach we adopted was by replacing the encoder in the second MAE-based method with CLIP’s image encoder while keeping the decoder structure unchanged.

- **Point-prompted UNet Segmentation**  
  A UNet-based model that incorporates **user-provided point prompts** (converted into heatmaps) to guide the segmentation process interactively.
![Model Architecture](assets/unet_r.png)

## Example Results from our models



## Installation

> **Requirements:**  
> The code requires **Python >= 3.8**, as well as **PyTorch >= 1.7** and **TorchVision >= 0.8**.  
> Please follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install both dependencies.  
> Installing **PyTorch and TorchVision with CUDA support** is *strongly recommended* for optimal performance.

Clone the repository:

```bash
git clone https://github.com/yourusername/point-prompted-segmentation.git
cd point-prompted-segmentation
```

## Getting Started

---

## Web Demo for Point-prompted Segmentation

The `Point-prompted-Segmentation` folder contains the UI application and guidance for the online version.  

For setup and usage instructions, please refer to [`Point-prompted-Segmentation/README.md`](Point-prompted-Segmentation/README.md).

## Model Checkpoint

Three versions of the model are available, each with a different **backbone size**. These can be loaded using the following code:

```python
from segment_anything import sam_model_registry

sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```
Click the links below to download the checkpoint for each model type:

-  [Download ViT-H (default)](https://example.com/vit_h_checkpoint.pth)
-  [Download ViT-L](https://example.com/vit_l_checkpoint.pth)
-  [Download ViT-B](https://example.com/vit_b_checkpoint.pth)

## Dataset

The dataset used in this project is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains images and pixel-level annotations for 37 categories of pet breeds.

We also provide our **preprocessed version** of the dataset, which can be downloaded from the following Google Drive link:

🔗 [Download Preprocessed Dataset (Google Drive)](https://drive.google.com/your-preprocessed-data-link)

## Contributors
s2102597 and s2091784

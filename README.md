# CV Mini Project: Point-prompted Segmentation

---

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

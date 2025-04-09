# Define the Dataset class
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import kornia.augmentation as K
from torch import nn
import random

# Define COLOR_JITTER augmentation that only applies to images
COLOR_JITTER = K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5)
# Define RandomHorizontalFlip augmentation
RANDOM_HORIZONTAL_FLIP = K.RandomHorizontalFlip(p=0.5)
# Define RandomVerticalFlip augmentation
RANDOM_VERTICAL_FLIP = K.RandomVerticalFlip(p=0.5)
# Define RandomRotation augmentation
RANDOM_ROTATION = K.RandomRotation(degrees=30.0, p=0.5)
# Define RandomAffine augmentation
RANDOM_AFFINE = K.RandomAffine(degrees=30, translate=(0.3, 0.3), p=0.5)
# Define RandomElasticTransform augmentation
RANDOM_ELASTIC_TRANSFORM = K.RandomElasticTransform(p=0.5)
# Define RandomCrop augmentation
RANDOM_CROP = K.RandomCrop(size=(128, 128), p=0.5)

# Define different augmentation functions (apply one per sample)
AUGMENTATIONS = [
    "RANDOM_HORIZONTAL_FLIP",
    "RANDOM_VERTICAL_FLIP",
    "RANDOM_ROTATION",
    "RANDOM_AFFINE",
    "RANDOM_ELASTIC_TRANSFORM",
    "RANDOM_CROP",
    "COLOR_JITTER"
]

class SegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to dataset root (e.g., 'new_dataset')
            split (str): "train" or "test"
            transform (callable, optional): Kornia augmentation transforms.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform  # Kornia transformations
        self.augmentations_list = AUGMENTATIONS  # Store available augmentations

        # Define directories
        self.image_dir = os.path.join(root_dir, split, "image")
        self.mask_dir = os.path.join(root_dir, split, "mask")

        self.image_paths = []
        self.mask_paths = []

        for category in ["cat", "dog"]:
            category_img_dir = os.path.join(self.image_dir, category)
            category_mask_dir = os.path.join(self.mask_dir, category)

            if not os.path.exists(category_img_dir) or not os.path.exists(category_mask_dir):
                raise FileNotFoundError(f"Missing category directory: {category_img_dir} or {category_mask_dir}")

            for breed in os.listdir(category_img_dir):
                breed_img_dir = os.path.join(category_img_dir, breed)
                breed_mask_dir = os.path.join(category_mask_dir, breed)

                if not os.path.exists(breed_mask_dir):
                    raise FileNotFoundError(f"Mask directory missing for {breed}")

                image_files = sorted(os.listdir(breed_img_dir))
                mask_files = sorted(os.listdir(breed_mask_dir))

                for img_file in image_files:
                    mask_file = img_file  # Mask should have the same filename
                    img_path = os.path.join(breed_img_dir, img_file)
                    mask_path = os.path.join(breed_mask_dir, mask_file)

                    if mask_file in mask_files:
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    else:
                        print(f"⚠️ Warning: Mask file {mask_file} not found for {img_file}")

        # print(f"✅ Loaded {len(self.image_paths)} samples from {self.split} dataset")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load `.npy` image and mask
        image = np.load(self.image_paths[idx]).astype(np.float32)  # Ensure float32
        mask = np.load(self.mask_paths[idx]).astype(np.int64)  # Ensure int64 (segmentation labels)

        # Get the category and breed from the path
        # Category is the parent directory of the image, e.g., new_dataset\train\image\cat\Egyptian_Mau\Egyptian_Mau_119.npy, category is cat
        category = self.image_paths[idx].split(os.sep)[-3]
        # Breed is the parent directory of the image, e.g., new_dataset\train\image\cat\Egyptian_Mau\Egyptian_Mau_119.npy, breed is Egyptian_Mau
        breed = self.image_paths[idx].split(os.sep)[-2]
        # Create text description combining category
        text_description = f"{category}"
        # print(f"Category: {category}")

        # Normalize image (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        # Convert mask from (H, W, 1) to (1, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        # Apply resizing to image and mask to (224, 224)
        image = nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear",
                                          align_corners=False).squeeze(0)
        mask = nn.functional.interpolate(mask.unsqueeze(0).float(), size=(224, 224), mode="nearest").squeeze(0)

        # Apply Kornia geometric augmentations (jointly to image & mask)
        if self.transform is not None:
            # Select a random augmentation
            random_aug_name = random.choice(self.augmentations_list)  # Include ColorJitter
            # print(f"Applying augmentation: {random_aug_name}")
            random_aug = globals()[random_aug_name]

            if random_aug_name == "COLOR_JITTER":
                # Apply ColorJitter only to the image
                image = COLOR_JITTER(image.unsqueeze(0)).squeeze(0)
            else:
                # Apply the random augmentation to image and mask
                # Ensure the mask is expanded to 3 channels for joint augmentation
                mask = mask.expand(3, -1, -1)
                # Concatenate the image and mask for joint augmentation, (3, H, W) and (3, H, W) to (6, H, W)
                image_mask = torch.cat([image, mask], dim=0)
                # Apply the random augmentation
                image_mask = random_aug(image_mask.unsqueeze(0)).squeeze(0)
                # Split the image and mask
                image, mask = image_mask[:3], image_mask[3:]
                # Ensure the mask is reduced back to 1 channel
                mask = mask[0].unsqueeze(0)

                if random_aug_name == "RANDOM_CROP":
                    # Resize the image and mask back to (224, 224)
                    image = nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
                    mask = nn.functional.interpolate(mask.unsqueeze(0).float(), size=(224, 224), mode="nearest").squeeze(0)

        # Convert the image from C, H, W to H, W, C
        image = image.permute(1, 2, 0)
        # Convert the mask from (1, H, W) to (H, W, 1)
        mask = mask.permute(1, 2, 0)
        # Convert the mask to (H, W)
        mask = mask.squeeze(2)

        # If the category is dog, set the object class to 1
        # If the category is cat, set the object class to 2
        if category == "dog":
            mask[mask > 0] = 1
            mask[mask <= 0] = 0
            # # Check if the mask has values other than 0 and 1
            # if len(np.unique(mask)) > 2:
            #     print(f"Unique values in mask: {np.unique(mask)}")
        elif category == "cat":
            mask[mask > 0] = 2
            mask[mask <= 0] = 0
            # # Check if the mask has values other than 0 and 2
            # if len(np.unique(mask)) > 2:
            #     print(f"Unique values in mask: {np.unique(mask)}")

        # Convert the mask to one-hot encoding, shape (H, W, 3), 0 is background, 1 is dog, 2 is cat
        mask = torch.nn.functional.one_hot(mask.to(torch.int64), num_classes=3).permute(2, 0, 1).float()

        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

        # Return the text description for CLIP training
        return image, mask, text_description



# Test the dataset
# if __name__ == "__main__":
#     # Create dataset instances
#     train_dataset = SegDataset(root_dir="new_dataset", split="train", transform=True) # Apply augmentations
#     print("The length of the dataset is: ", len(train_dataset))
#     # test_dataset = SegDataset(root_dir="new_dataset", split="test", transform=None)  # No augmentation for test set
#     # print("The length of the dataset is: ", len(test_dataset))
#
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
#     # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)
#
#     # Check sample output
#     for images, masks, text_description in train_loader:
#         print("Batch Image shape:", images.shape)  # Expected: (B, H, W, C)
#         print("Batch Mask shape:", masks.shape)  # Expected: (B, C, H, W)
#         print("Batch text_description:", text_description)
#         print("Unique values in mask:", torch.unique(masks[0]))  # Expected: 0, 1, 2
#
#         # Display a sample image and mask with color bar
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(images[0])
#         plt.title("Image")
#         plt.colorbar()
#         plt.axis("off")
#         plt.subplot(1, 2, 2)
#         plt.imshow(masks[0].permute(1, 2, 0)[:, :, 0])
#         plt.title("Mask")
#         plt.colorbar()
#         plt.axis("off")
#         plt.show()
#         break

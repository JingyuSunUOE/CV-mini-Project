import torch
from dataset import SegDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
from skimage.util import random_noise
import random

from models.Unet.unet import UNetSeg

def compute_mean_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute mean Dice score over a batch of predictions and ground truth masks.

    Args:
        preds (torch.Tensor): Predicted masks of shape [B, C, H, W], values in [0, 1] or binary.
        targets (torch.Tensor): Ground truth masks of same shape, binary.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Mean Dice score across batch and channels.
    """
    if preds.dtype != torch.float32:
        preds = preds.float()
    if targets.dtype != torch.float32:
        targets = targets.float()

    # Flatten: [B, C, H, W] -> [B, C, H*W]
    preds = preds.view(preds.shape[0], preds.shape[1], -1)
    targets = targets.view(targets.shape[0], targets.shape[1], -1)

    intersection = (preds * targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection + eps) / (union + eps)  # shape [B, C]
    mean_dice = dice.mean().item()

    return mean_dice

def apply_gaussian_pixel_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    """
    Apply Gaussian pixel noise assuming input is [0, 1] float32, and std is in [0, 255] range.
    Returns float32 tensor still in [0, 1].
    """
    if images.dtype != torch.float32:
        images = images.float()

    # Scale to 0–255 to add noise properly
    images_255 = images * 255.0
    noise = torch.randn_like(images_255) * std
    noisy_images = images_255 + noise
    noisy_images = torch.clamp(noisy_images, 0, 255) / 255.0  # back to [0, 1]

    return noisy_images

def evaluate_model_with_gaussian_noise(model, dataloader, std, device):
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Add Gaussian noise
            noisy_images = apply_gaussian_pixel_noise(images, std=std)

            # Model Inference
            outputs = model(noisy_images)
            outputs = (outputs > 0.5).float()

            # Dice calculation
            dice = compute_mean_dice(outputs, masks)
            print(
                f"std={std}, batch {count + 1}: Mean Dice score: {dice:.4f}")
            total_dice += dice
            count += 1

    return total_dice / count

def apply_gaussian_blur(images: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    Apply iterative 3x3 Gaussian blur to a batch of images.

    Parameters:
        images (torch.Tensor): Input images of shape [B, 3, H, W], values in [0, 1].
        iterations (int): Number of times to apply the blur (0~9).

    Returns:
        torch.Tensor: Blurred images, same shape, values in [0, 1].
    """
    if images.dim() == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    device = images.device

    # Define 3x3 Gaussian kernel
    kernel = torch.tensor([
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]
    ], device=device)
    kernel = kernel / 16.0  # Normalize
    kernel = kernel.view(1, 1, 3, 3)  # shape: [1, 1, 3, 3]
    kernel = kernel.repeat(3, 1, 1, 1)  # shape: [3, 1, 3, 3] → for RGB channels

    # Pad images for "same" output size
    blurred = images
    for _ in range(iterations):
        blurred = F.conv2d(blurred, kernel, padding=1, groups=3)

    blurred = blurred.permute(0, 2, 3, 1)
    return blurred


def evaluate_model_with_blur(model, dataloader, iterations, device):
    """
    Evaluate model on blurred images using given number of blur iterations.

    Args:
        model: segmentation model
        dataloader: test dataloader
        iterations: number of times to apply the 3x3 Gaussian blur (0~9)
        device: torch.device

    Returns:
        float: mean Dice score on the full test set
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            blurred_images = apply_gaussian_blur(images, iterations=iterations)

            outputs = model(blurred_images)
            outputs = (outputs > 0.5).float()

            dice = compute_mean_dice(outputs, masks)
            print(
                f"Blur Iteration = {iterations}, batch {count + 1}: Mean Dice score: {dice:.4f}")
            total_dice += dice
            count += 1

    return total_dice / count

def apply_contrast_increase(images: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Apply contrast increase by multiplying pixel values by a factor.
    Assumes input in [0, 1], returns output in [0, 1].
    """
    if images.dtype != torch.float32:
        images = images.float()

    # Convert to 0-255 space
    images_255 = images * 255.0
    contrast_images = images_255 * factor
    contrast_images = torch.clamp(contrast_images, 0, 255.0)

    # Convert back to [0, 1]
    return contrast_images / 255.0

def evaluate_model_with_contrast(model, dataloader, factor, device):
    """
    Evaluate the model on the test dataset with contrast-enhanced images.

    Args:
        model: The segmentation model.
        dataloader: Test dataloader.
        factor (float): Contrast multiplication factor (e.g., 1.05, 1.2).
        device: torch.device, e.g. "cuda" or "cpu".

    Returns:
        float: Mean Dice score over the test set.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Apply contrast-enhancing perturbation
            contrast_images = apply_contrast_increase(images, factor=factor)

            # Model prediction and Dice calculation
            outputs = model(contrast_images)
            outputs = (outputs > 0.5).float()
            dice = compute_mean_dice(outputs, masks)
            print(f"Contrast Factor = {factor}, batch {count + 1}: Mean Dice score: {dice:.4f}")

            total_dice += dice
            count += 1

    return total_dice / count

def apply_contrast_decrease(images: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Apply contrast decrease to a batch of images by multiplying each pixel by a factor.
    Assumes input in [0, 1], returns output in [0, 1].

    Args:
        images (torch.Tensor): [B, 3, H, W] or [B, H, W, 3], values in [0, 1]
        factor (float): Contrast decrease factor (e.g., 0.95, 0.9, ..., 0.1)

    Returns:
        torch.Tensor: Contrast-decreased images, same shape, values in [0, 1]
    """
    if images.dtype != torch.float32:
        images = images.float()

    images_255 = images * 255.0
    contrast_images = images_255 * factor
    contrast_images = torch.clamp(contrast_images, 0, 255.0)

    return contrast_images / 255.0

def evaluate_model_with_contrast_decrease(model, dataloader, factor, device):
    """
    Evaluate the model on contrast-decreased images.

    Args:
        model: The segmentation model.
        dataloader: Test dataloader.
        factor (float): Multiplicative factor for contrast decrease.
        device: torch.device

    Returns:
        float: Mean Dice score over the test set.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            decreased_images = apply_contrast_decrease(images, factor)

            outputs = model(decreased_images)
            outputs = (outputs > 0.5).float()

            dice = compute_mean_dice(outputs, masks)
            print(f"Contrast Factor = {factor}, batch {count + 1}: Mean Dice score: {dice:.4f}")
            total_dice += dice
            count += 1

    return total_dice / count

def apply_brightness_increase(images: torch.Tensor, offset: float) -> torch.Tensor:
    """
    Increase image brightness by adding a constant offset to each pixel.

    Args:
        images (torch.Tensor): Input tensor [B, 3, H, W] or [B, H, W, 3], values in [0, 1].
        offset (float): Value to add to each pixel (e.g., 5, 10, ..., 45).

    Returns:
        torch.Tensor: Brightness-increased images, same shape, values in [0, 1].
    """
    if images.dtype != torch.float32:
        images = images.float()

    images_255 = images * 255.0
    bright_images = images_255 + offset
    bright_images = torch.clamp(bright_images, 0, 255.0)

    return bright_images / 255.0

def evaluate_model_with_brightness_increase(model, dataloader, offset, device):
    """
    Evaluate the model on brightness-increased images.

    Args:
        model: The segmentation model.
        dataloader: Test dataloader.
        offset (float): Brightness offset to add (e.g., 10, 20, ...)
        device: torch.device

    Returns:
        float: Mean Dice score over the test set.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            bright_images = apply_brightness_increase(images, offset)

            outputs = model(bright_images)
            outputs = (outputs > 0.5).float()

            dice = compute_mean_dice(outputs, masks)
            print(f"Brightness Offset = {offset}, batch {count + 1}: Mean Dice score: {dice:.4f}")
            total_dice += dice
            count += 1

    return total_dice / count


def apply_brightness_decrease(images: torch.Tensor, offset: float) -> torch.Tensor:
    """
    Decrease image brightness by subtracting a constant offset from each pixel.

    Args:
        images (torch.Tensor): Input tensor [B, 3, H, W] or [B, H, W, 3], values in [0, 1].
        offset (float): Value to subtract from each pixel (e.g., 5, 10, ..., 45).

    Returns:
        torch.Tensor: Brightness-decreased images, same shape, values in [0, 1].
    """
    if images.dtype != torch.float32:
        images = images.float()

    images_255 = images * 255.0
    dark_images = images_255 - offset
    dark_images = torch.clamp(dark_images, 0, 255.0)

    return dark_images / 255.0

def evaluate_model_with_brightness_decrease(model, dataloader, offset, device):
    """
    Evaluate the model on brightness-decreased images.

    Args:
        model: The segmentation model.
        dataloader: Test dataloader.
        offset (float): Brightness decrease offset (e.g., 5, 10, ...)
        device: torch.device

    Returns:
        float: Mean Dice score over the test set.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

            dark_images = apply_brightness_decrease(images, offset)

            outputs = model(dark_images)
            outputs = (outputs > 0.5).float()

            dice = compute_mean_dice(outputs, masks)
            total_dice += dice
            count += 1

    return total_dice / count

def apply_salt_and_pepper_noise(images: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Apply salt and pepper noise to a batch of images.

    Args:
        images (torch.Tensor): [B, H, W, 3], values in [0, 1], float32.
        amount (float): Noise strength, e.g., 0.02 means 2% of pixels are flipped.

    Returns:
        torch.Tensor: Noised images, same shape, values in [0, 1], float32.
    """
    if images.dtype != torch.float32:
        images = images.float()

    if images.shape[-1] == 3:  # [B, H, W, 3] → [B, 3, H, W]
        images = images.permute(0, 3, 1, 2)

    noisy_batch = []
    for img in images:
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        noisy_np = random_noise(img_np, mode='s&p', amount=amount, clip=True)
        noisy_np = noisy_np.astype(np.float32)
        noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1)  # [C, H, W]
        noisy_batch.append(noisy_tensor)

    return torch.stack(noisy_batch).to(images.device).permute(0, 2, 3, 1)

def evaluate_model_with_sp_noise(model, dataloader, amount, device):
    """
    Evaluate the model on salt and pepper noised images.

    Args:
        model: The segmentation model.
        dataloader: Test dataloader.
        amount (float): Noise level (e.g., 0.02, 0.1)
        device: torch.device

    Returns:
        float: Mean Dice score over the test set.
    """
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            noisy_images = apply_salt_and_pepper_noise(images, amount)

            outputs = model(noisy_images)
            outputs = (outputs > 0.5).float()
            dice = compute_mean_dice(outputs, masks)
            print(f"Salt & Pepper Noise Level = {amount}, batch {count + 1}: Mean Dice score: {dice:.4f}")

            total_dice += dice
            count += 1

    return total_dice / count

def apply_random_occlusion(images: torch.Tensor, masks: torch.Tensor, square_size: int):
    """
    Apply random square occlusion (black square) on images and zero out the corresponding mask region.

    Args:
        images (torch.Tensor): [B, H, W, 3], pixel values in [0, 1]
        masks (torch.Tensor): [B, C, H, W], binary masks
        square_size (int): Size of the black square edge (0 for no occlusion)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (occluded_images, updated_masks)
    """
    if images.dtype != torch.float32:
        images = images.float()
    if masks.dtype != torch.float32:
        masks = masks.float()

    B, H, W, C = images.shape
    occluded_images = images.clone()
    updated_masks = masks.clone()

    for i in range(B):
        if square_size == 0:
            continue  # no occlusion

        # Randomly select the coordinates of the upper left corner of a block (x, y)
        x = random.randint(0, max(0, W - square_size))
        y = random.randint(0, max(0, H - square_size))

        # Mask the image (note: the channel is at the end)
        occluded_images[i, y:y+square_size, x:x+square_size, :] = 0.0

        # Update mask synchronously (channels in the second dimension)
        updated_masks[i, :, y:y+square_size, x:x+square_size] = 0.0

    return occluded_images, updated_masks

def evaluate_model_with_occlusion(model, dataloader, square_size, device):
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            occluded_images, updated_masks = apply_random_occlusion(images, masks, square_size)

            outputs = model(occluded_images)
            outputs = (outputs > 0.5).float()
            dice = compute_mean_dice(outputs, updated_masks)
            print(f"Occlusion Size = {square_size}, batch {count + 1}: Mean Dice score: {dice:.4f}")

            total_dice += dice
            count += 1

    return total_dice / count

if __name__ == "__main__":

    # Load the test dataset
    test_dataset = SegDataset(root_dir="new_dataset", split="test", transform=None)  # No augmentation for test set
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True,
                            num_workers=8, pin_memory=True, drop_last=True,
                            prefetch_factor=6, persistent_workers=True
                            )
    print("The number of test samples:", len(test_dataset))
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load the model
    # # ----------------------------------------------------------
    # model = UNetSeg()
    # parameter_path = "logs/Unet/best_param.pth"
    # state_dict = torch.load(parameter_path)
    # # remove the 'module' prefix
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)
    # # ----------------------------------------------------------
    #
    # # move to device and enable multi-GPU
    # model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs.")
    #     model = torch.nn.DataParallel(model)
    #
    # model.eval()
    #
    # print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    #
    # # ----------------------------------------------------------
    # sample_path = "samples/Robustness_test"
    # os.makedirs(sample_path, exist_ok=True)  # Ensure the directory exists
    #
    # # Evaluate the model on different std levels
    # std_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    # mean_dice_list_gaussian = []
    #
    # for std in std_list:
    #     print("Std Level:", std)
    #     mean_dice = evaluate_model_with_gaussian_noise(model, test_loader, std, device)
    #     print(f"[std={std}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_gaussian.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different std levels: {mean_dice_list_gaussian}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_gaussian = np.array(mean_dice_list_gaussian)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_gaussian.txt"), mean_dice_list_gaussian)
    # # ----------------------------------------------------------
    #
    # blur_iterations = list(range(10))  # 0 to 9
    # mean_dice_list_blur = []
    #
    # for i in blur_iterations:
    #     print("Blur Iteration:", i)
    #     mean_dice = evaluate_model_with_blur(model, test_loader, iterations=i, device=device)
    #     print(f"[Blur Iteration = {i}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_blur.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different blur iterations: {mean_dice_list_blur}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_blur = np.array(mean_dice_list_blur)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_blur.txt"), mean_dice_list_blur)
    # # ----------------------------------------------------------
    #
    # contrast_factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
    # mean_dice_list_contrast = []
    #
    # for factor in contrast_factors:
    #     print("Contrast Factor:", factor)
    #     mean_dice = evaluate_model_with_contrast(model, test_loader, factor=factor, device=device)
    #     print(f"[Contrast Factor = {factor}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_contrast.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different contrast factors: {mean_dice_list_contrast}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_contrast = np.array(mean_dice_list_contrast)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_contrast.txt"), mean_dice_list_contrast)
    # # ----------------------------------------------------------
    #
    # contrast_decrease_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    # mean_dice_list_contrast_decrease = []
    #
    # for factor in contrast_decrease_factors:
    #     print("Contrast Factor:", factor)
    #     mean_dice = evaluate_model_with_contrast_decrease(model, test_loader, factor, device)
    #     print(f"[Contrast ↓ Factor = {factor}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_contrast_decrease.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different contrast factors: {mean_dice_list_contrast_decrease}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_contrast_decrease = np.array(mean_dice_list_contrast_decrease)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_contrast_decrease.txt"), mean_dice_list_contrast_decrease)
    # # ----------------------------------------------------------
    #
    # brightness_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # mean_dice_list_brightness = []
    #
    # for offset in brightness_offsets:
    #     print("Brightness Offset:", offset)
    #     mean_dice = evaluate_model_with_brightness_increase(model, test_loader, offset, device)
    #     print(f"[Brightness Offset = {offset}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_brightness.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different brightness offsets: {mean_dice_list_brightness}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_brightness = np.array(mean_dice_list_brightness)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_brightness.txt"), mean_dice_list_brightness)
    # # ----------------------------------------------------------
    #
    # brightness_decrease_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # mean_dice_list_brightness_decrease = []
    #
    # for offset in brightness_decrease_offsets:
    #     print("Brightness -", offset)
    #     mean_dice = evaluate_model_with_brightness_decrease(model, test_loader, offset, device)
    #     print(f"[Brightness -{offset}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_brightness_decrease.append(mean_dice)
    #
    # print(
    #     f"Mean Dice scores for different brightness offsets: {mean_dice_list_brightness_decrease}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_brightness_decrease = np.array(mean_dice_list_brightness_decrease)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_brightness_decrease.txt"), mean_dice_list_brightness_decrease)
    # # ----------------------------------------------------------
    #
    # sp_amounts = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    # mean_dice_list_sp = []
    #
    # for amt in sp_amounts:
    #     print("Salt & Pepper noise:", amt)
    #     mean_dice = evaluate_model_with_sp_noise(model, test_loader, amount=amt, device=device)
    #     print(f"[Salt & Pepper noise {amt:.2f}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_sp.append(mean_dice)
    #
    # print(
    #     f"Mean Dice scores for different sp amounts: {mean_dice_list_sp}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_sp = np.array(mean_dice_list_sp)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_sp.txt"), mean_dice_list_sp)
    # # ----------------------------------------------------------
    #
    # occlusion_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # mean_dice_list_occlusion = []
    #
    # for size in occlusion_sizes:
    #     print("Occlusion Size:", size)
    #     mean_dice = evaluate_model_with_occlusion(model, test_loader, size, device)
    #     print(f"[Occlusion Size = {size}] Mean Dice: {mean_dice:.4f}")
    #     mean_dice_list_occlusion.append(mean_dice)
    # print(
    #     f"Mean Dice scores for different occlusion sizes: {mean_dice_list_occlusion}")
    # print("-------------------------------------------------------------")
    #
    # # Save the mean_dice_list to txt
    # mean_dice_list_occlusion = np.array(mean_dice_list_occlusion)
    # np.savetxt(os.path.join(sample_path, "mean_dice_list_occlusion.txt"), mean_dice_list_occlusion)
    # # ----------------------------------------------------------

    # Evaluate the model
    with torch.no_grad():
        num_batches = 0
        std_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        iterations_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        contrast_factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
        contrast_decrease_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
        brightness_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        brightness_decrease_offsets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        sp_amounts = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
        occlusion_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

        for i, (images, masks, text_description) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            print("images shape:", images.shape)
            print("masks shape:", masks.shape)

            # noisy_images = apply_gaussian_pixel_noise(images, std=18)
            # noisy_images = apply_gaussian_blur(images, iterations=9)
            # noisy_images = apply_contrast_increase(images, factor=1.25)
            # noisy_images = apply_contrast_decrease(images, factor=0.60)
            # noisy_images = apply_brightness_increase(images, offset=45)
            # noisy_images = apply_brightness_decrease(images, offset=45)
            noisy_images = apply_salt_and_pepper_noise(images, amount=0.18)
            # noisy_images, masks = apply_random_occlusion(images, masks, square_size=40)
            print("noisy_images shape:", noisy_images.shape)

            # Plot the images and noisy_images
            import matplotlib.pyplot as plt

            plot_image = images[0].cpu().numpy()
            plot_noisy_image = noisy_images[0].cpu().numpy()

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(plot_image)
            axs[0].set_title("Original Image")
            axs[1].imshow(plot_noisy_image)
            axs[1].set_title("Noisy Image")
            axs[2].imshow(masks[0].permute(1, 2, 0).cpu().numpy())
            axs[2].set_title("Ground Truth")
            plt.show()

            break

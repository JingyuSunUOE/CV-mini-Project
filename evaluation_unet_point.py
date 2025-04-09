import torch
from dataset import SegDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import os
import random
from PIL import ImageDraw
import numpy as np

from models.Unet_Point.unet_point import generate_click_and_heatmap_binary
from models.Unet_Point.unet_point import UNetPointSeg


def compute_iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def compute_dice(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean()

def compute_pixel_accuracy(pred, target):
    correct = (pred == target).sum().float()
    total = target.numel()
    return correct / total

def compute_precision(pred, target, smooth=1e-6):
    TP = (pred * target).sum(dim=(1, 2, 3))
    FP = (pred * (1 - target)).sum(dim=(1, 2, 3))
    precision = (TP + smooth) / (TP + FP + smooth)
    return precision.mean()

def compute_recall(pred, target, smooth=1e-6):
    TP = (pred * target).sum(dim=(1, 2, 3))
    FN = ((1 - pred) * target).sum(dim=(1, 2, 3))
    recall = (TP + smooth) / (TP + FN + smooth)
    return recall.mean()

def evaluate_segmentation(output, mask, threshold=0.5):
    """
    Evaluate segmentation metrics for binary masks
    Args:
        output (Tensor): model logits (B, 1, H, W)
        mask (Tensor): ground truth binary mask (B, 1, H, W)
    Returns:
        dict of evaluation results
    """
    output = torch.sigmoid(output)
    pred = (output > threshold).float()

    iou = compute_iou(pred, mask)
    dice = compute_dice(pred, mask)
    pixel_acc = compute_pixel_accuracy(pred, mask)
    precision = compute_precision(pred, mask)
    recall = compute_recall(pred, mask)

    return {
        'IoU': iou.item(),
        'Dice': dice.item(),
        'Pixel Accuracy': pixel_acc.item(),
        'Precision': precision.item(),
        'Recall': recall.item()
    }

# Test
if __name__ == "__main__":
    # Load the model
    model = UNetPointSeg()
    parameters_path = "logs/Unet_Point/best_param.pth"
    state_dict = torch.load(parameters_path)
    # remove the 'module' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    # ----------------------------------------------------------
    # pretrained_dict = "logs/Unet_Point/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # Set the device
    # device_ids = [0, 1]  # Use multiple GPUs, the maximum GPU number of the server is 8 H100
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use single GPU

    model.to(device)

    print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load the test dataset
    test_dataset = SegDataset(root_dir="new_dataset", split="test", transform=None)  # No augmentation for test set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    print("The number of test samples:", len(test_dataset))

    model.eval()

    sample_path = "samples/Unet_Point"

    os.makedirs(sample_path, exist_ok=True)  # Ensure the directory exist

    # Evaluate the model
    with torch.no_grad():
        # Initialize cumulative metrics
        cumulative_metrics = {'IoU': 0, 'Dice': 0, 'Pixel Accuracy': 0, 'Precision': 0, 'Recall': 0}
        num_batches = 0

        saved_metrics = []

        for i, (images, masks, text_description) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            # print("images shape:", images.shape)
            # print("masks shape:", masks.shape)

            # ----------------------------------------------------------
            heatmaps = []
            updated_masks = []
            click_points = []
            for j in range(images.shape[0]):
                click_point, heatmap, updated_mask = generate_click_and_heatmap_binary(masks[j].permute(1,2,0).cpu().detach().numpy())
                heatmaps.append(heatmap)
                updated_masks.append(updated_mask)
                click_points.append(click_point)

            heatmaps = np.array(heatmaps)
            heatmaps = torch.from_numpy(heatmaps).float()  # to float32 (B, 224, 224)
            # Convert to (B, 224, 224, 1)
            heatmaps = heatmaps.unsqueeze(-1)
            heatmaps = heatmaps.to(device)

            updated_masks = np.array(updated_masks)
            updated_masks = torch.from_numpy(updated_masks)
            updated_masks = updated_masks.to(device) # (B, 224, 224)
            updated_masks = updated_masks.unsqueeze(1) # (B, 1, 224, 224)

            image_combined = torch.cat((images, heatmaps), dim=-1)

            outputs = model(image_combined) # Forward pass (B, 1, 224, 224)

            metrics = evaluate_segmentation(outputs, updated_masks, threshold=0.5)
            print(f"Batch {i + 1}: {metrics}")

            # Randomly select a sample from the batch
            idx = random.randint(0, images.size(0) - 1)

            image = images[idx].cpu()
            mask = updated_masks[idx].cpu() # (1, 224, 224)
            # print("the max value of mask:", mask.max())
            # print("the min value of mask:", mask.min())
            # print("Number of foreground pixels:", mask.sum().item())
            outputs = torch.sigmoid(outputs)
            pred_mask = outputs[idx].cpu()

            # Convert tensors to PIL images
            image_pil = ToPILImage()(image.permute(2, 0, 1).float())  # RGB
            mask_vis = (mask.squeeze(0) * 255).byte()  # to uint8
            mask_pil = ToPILImage()(mask_vis)
            pred_mask_pil = ToPILImage()(pred_mask.squeeze(0))

            # Get user-clicked point (x, y) from heatmap
            click_coord = click_points[idx]

            # Function to draw a red dot on an image at the specified (x, y) location
            def draw_click_point(pil_img, coord, color="red", radius=4):
                draw = ImageDraw.Draw(pil_img)
                x, y = coord
                left_up_point = (x - radius, y - radius)
                right_down_point = (x + radius, y + radius)
                draw.ellipse([left_up_point, right_down_point], fill=color)


            # Draw the click point on all three images
            draw_click_point(image_pil, click_coord)
            draw_click_point(mask_pil, click_coord)
            draw_click_point(pred_mask_pil, click_coord)

            # Save the images
            image_pil.save(os.path.join(sample_path, f"batch_{i + 1}_image.jpg"))
            mask_pil.save(os.path.join(sample_path, f"batch_{i + 1}_gt_mask.png"))
            pred_mask_pil.save(os.path.join(sample_path, f"batch_{i + 1}_pred_mask.png"))

            # Accumulate the metrics
            for key in cumulative_metrics:
                cumulative_metrics[key] += metrics[key]
            num_batches += 1

        # Calculate and print the average metrics
        average_metrics = {key: value / num_batches for key, value in cumulative_metrics.items()}
        print("Average Metrics:", average_metrics)
        saved_metrics.append(average_metrics)

    # Save the metrics list to txt
    saved_metrics_path = os.path.join(sample_path, "saved_metrics.txt")
    with open(saved_metrics_path, "w") as f:
        for metrics in saved_metrics:
            f.write(f"{metrics}\n")
    print(f"Metrics saved to {saved_metrics_path}")

import torch
from dataset import SegDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import os
import random
import sys

from models.Unet.unet import UNetSeg
import models.MAE.mae_seg as mae_seg
import models.MAE.mae as mae
import models.MAE.models_mae as models_mae
import models.MAE.mae_mask as mae_mask
import models.clip.clip as clip
import models.clip.clip_mask as clip_mask
import models.clip.clip_model as clip_model
import models.clip.lseg as lseg

def compute_iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target - pred * target).sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def compute_mean_iou_per_class(pred, target, smooth=1e-6):
    """
    Compute mean IoU over classes.
    Args:
        pred (Tensor): (B, C, H, W), binary or soft prediction after thresholding
        target (Tensor): (B, C, H, W), ground truth one-hot
    Returns:
        mean IoU over all classes
    """
    num_classes = pred.shape[1]
    ious = []
    for cls in range(num_classes):
        pred_cls = pred[:, cls, :, :]
        target_cls = target[:, cls, :, :]
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = (pred_cls + target_cls - pred_cls * target_cls).sum(dim=(1, 2))
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)  # shape: [B]

    # Stack ious into shape (num_classes, B), then mean over both axes
    ious = torch.stack(ious, dim=0)  # (C, B)
    mean_iou = ious.mean()
    return mean_iou

def compute_dice(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return dice.mean()

def compute_pixel_accuracy(pred, target):
    pred_class = pred.argmax(dim=1)  # (B, H, W)
    target_class = target.argmax(dim=1)  # (B, H, W)
    correct = (pred_class == target_class).sum().float()
    total = target_class.numel()
    return correct / total

def compute_precision(pred, target, smooth=1e-6):
    pred_class = pred.argmax(dim=1)  # (B, H, W)
    target_class = target.argmax(dim=1)  # (B, H, W)

    TP = (pred_class == target_class).float() * (target_class != 0).float()  # Only count non-background categories
    FP = (pred_class != target_class).float() * (pred_class != 0).float()  # Types of false positives
    precision = (TP.sum() + smooth) / (TP.sum() + FP.sum() + smooth)
    return precision

def compute_recall(pred, target, smooth=1e-6):
    pred_class = pred.argmax(dim=1)  # (B, H, W)
    target_class = target.argmax(dim=1)  # (B, H, W)

    TP = (pred_class == target_class).float() * (target_class != 0).float()
    FN = (pred_class != target_class).float() * (target_class != 0).float()
    recall = (TP.sum() + smooth) / (TP.sum() + FN.sum() + smooth)
    return recall

def evaluate_segmentation(output, mask, threshold=0.5):
    """
    Evaluate the output of the segmentation model and calculate IoU, Dice, Pixel Accuracy, Precision and Recall
    Args:
        output (Tensor): model output (B, C, H, W), usually raw logits
        mask (Tensor): Ground truth mask (B, C, H, W), one-hot encoding
        threshold (float): Threshold used to binarize the probability map
    Returns:
        metrics (dict): a dictionary containing the values of each metric
    """
    output = torch.softmax(output, dim=1)
    pred = (output > threshold).float()

    iou = compute_iou(pred, mask)
    mean_iou = compute_mean_iou_per_class(pred, mask)
    dice = compute_dice(pred, mask)
    pixel_acc = compute_pixel_accuracy(pred, mask)
    precision = compute_precision(pred, mask)
    recall = compute_recall(pred, mask)

    return {
        'IoU': iou.item(),
        'Mean IoU': mean_iou.item(),
        'Dice': dice.item(),
        'Pixel Accuracy': pixel_acc.item(),
        'Precision': precision.item(),
        'Recall': recall.item()
    }


# Test
if __name__ == "__main__":

    # Load the model

    # ----------------------------------------------------------
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
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_finetune/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_retrain/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Tiny/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Tiny_finetune/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Tiny_retrain/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Large/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Large_finetune/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_seg'] = mae_seg
    # sys.modules['models.mae'] = mae
    # pretrained_dict = "logs/MAE_Seg_Large_retrain/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_mask'] = mae_mask
    # sys.modules['models.models_mae'] = models_mae
    # pretrained_dict = "logs/MAE_Mask_Seg_Vit_L/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_mask'] = mae_mask
    # sys.modules['models.models_mae'] = models_mae
    # pretrained_dict = "logs/MAE_Mask_Seg_Vit_B/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.mae_mask'] = mae_mask
    # sys.modules['models.models_mae'] = models_mae
    # pretrained_dict = "logs/MAE_Mask_Seg_Vit_B_retrain/best_model.pt"
    # model = torch.load(pretrained_dict,weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.clip'] = clip
    # sys.modules['models.clip_model'] = clip_model
    # sys.modules['models.lseg'] = lseg
    # pretrained_dict = "logs/CLIP_Seg_Base/best_model.pt"
    # model = torch.load(pretrained_dict, weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.clip'] = clip
    # sys.modules['models.clip_model'] = clip_model
    # sys.modules['models.lseg'] = lseg
    # pretrained_dict = "logs/CLIP_Seg_Vit_B_16/best_model.pt"
    # model = torch.load(pretrained_dict, weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.clip'] = clip
    # sys.modules['models.clip_model'] = clip_model
    # sys.modules['models.lseg'] = lseg
    # pretrained_dict = "logs/CLIP_Seg_Vit_B_32/best_model.pt"
    # model = torch.load(pretrained_dict, weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # sys.modules['models.clip'] = clip
    # sys.modules['models.clip_model'] = clip_model
    # sys.modules['models.lseg'] = lseg
    # pretrained_dict = "logs/CLIP_Seg_Vit_L_14/best_model.pt"
    # model = torch.load(pretrained_dict, weights_only=False)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    sys.modules['models.clip'] = clip
    sys.modules['models.clip_model'] = clip_model
    sys.modules['models.clip_mask'] = clip_mask
    pretrained_dict = "logs/CLIP_Mask_Seg_Vit_L_14/best_model.pt"
    model = torch.load(pretrained_dict, weights_only=False)
    # ----------------------------------------------------------

    # Set the device
    torch.cuda.init()
    device_ids = [0, 1]  # Use multiple GPUs, the maximum GPU number of the server is 8 H100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print("The number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load the test dataset
    test_dataset = SegDataset(root_dir="new_dataset", split="test", transform=None)  # No augmentation for test set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=24)
    print("The number of test samples:", len(test_dataset))

    model.eval()

    # sample_path = "samples/Unet"

    # sample_path = "samples/MAE_Seg"
    # sample_path = "samples/MAE_Seg_finetune"
    # sample_path = "samples/MAE_Seg_retrain"

    # sample_path = "samples/MAE_Seg_Tiny"
    # sample_path = "samples/MAE_Seg_Tiny_finetune"
    # sample_path = "samples/MAE_Seg_Tiny_retrain"

    # sample_path = "samples/MAE_Seg_Large"
    # sample_path = "samples/MAE_Seg_Large_finetune"
    # sample_path = "samples/MAE_Seg_Large_retrain"

    # sample_path = "samples/MAE_Mask_Seg_Vit_L"
    # sample_path = "samples/MAE_Mask_Seg_Vit_B"
    # sample_path = "samples/MAE_Mask_Seg_Vit_B_retrain"

    # sample_path = "samples/CLIP_Seg_Base"
    # sample_path = "samples/CLIP_Seg_Vit_B_16"
    # sample_path = "samples/CLIP_Seg_Vit_B_32"
    # sample_path = "samples/CLIP_Seg_Vit_L_14"

    sample_path = "samples/CLIP_Mask_Seg_Vit_L_14"

    os.makedirs(sample_path, exist_ok=True)  # Ensure the directory exists
    # Evaluate the model
    with torch.no_grad():
        # Initialize cumulative metrics
        cumulative_metrics = {'IoU': 0, 'Mean IoU':0, 'Dice': 0, 'Pixel Accuracy': 0, 'Precision': 0, 'Recall': 0}
        num_batches = 0

        saved_metrics = []

        for i, (images, masks, text_description) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            # print("images shape:", images.shape)
            # print("masks shape:", masks.shape)

            outputs = model(images)
            metrics = evaluate_segmentation(outputs, masks, threshold=0.5)
            print(f"Batch {i + 1}: {metrics}")

            # Randomly save one image, ground truth mask, and predicted mask
            idx = random.randint(0, images.size(0) - 1)  # Random index in the batch
            image = images[idx].cpu()
            mask = masks[idx].cpu()
            pred_mask = outputs[idx].argmax(dim=0).cpu()

            # Determine type based on unique pixel values in pred_mask
            pred_mask_classes = torch.unique(pred_mask)
            print("Unique Value:", pred_mask_classes)

            # If the unique values contain three classes, 0,1,2, choose the majority one (ignoring 0)
            if set(pred_mask_classes.tolist()) == {0, 1, 2}:
                # First make sure pred_mask.flatten() is not empty after removing background (0)
                foreground_mask = pred_mask[pred_mask != 0]  # Exclude background pixels
                if foreground_mask.numel() > 0:  # Ensure it's not empty
                    majority_pred_class = torch.mode(foreground_mask.flatten())[
                        0].item()  # Get the category with the most occurrences
                    print("Majority Class:", majority_pred_class)
                    # Using dictionary mappings for increased flexibility
                    class_mapping = {1: "dog", 2: "cat"}
                    pred_type = class_mapping.get(majority_pred_class,
                                                  "unknown")  # If there is no corresponding category, the default value is unknown
                else:
                    pred_type = "unknown"  # If no valid value is found, return the default category
            else:
                if 1 in pred_mask_classes:  # Dog
                    pred_type = "dog"
                elif 2 in pred_mask_classes:  # Cat
                    pred_type = "cat"
                else:
                    pred_type = "unknown"

            print("Prediction Type:", pred_type)
            print("------------------------------------------")

            # Convert image to [C, H, W] format
            image = ToPILImage()(image.permute(2, 0, 1).float())

            # Collapse one-hot mask (3, 224, 224) to single channel (224, 224)
            # mask = ToPILImage()(mask.argmax(dim=0).byte())
            mask = ToPILImage()(mask.argmax(dim=0).float())

            # Ensure pred_mask is single-channel for PIL conversion
            # pred_mask = ToPILImage()(pred_mask.byte())
            pred_mask = ToPILImage()(pred_mask.float())

            # Save the images with updated suffix for prediction
            image.save(os.path.join(sample_path, f"batch_{i + 1}_image.jpg"))
            mask.save(os.path.join(sample_path, f"batch_{i + 1}_gt_mask.png"))
            pred_mask.save(os.path.join(sample_path, f"batch_{i + 1}_pred_mask_{pred_type}.png"))

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

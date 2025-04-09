# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import models.clip as clip
from models.clip.lseg import LSegNet
import torch.nn.functional as F

from dataset import SegDataset
from tqdm import tqdm
import wandb
import os
import torch.nn as nn
from datetime import datetime
from early_stopping import EarlyStopping
import time
import sys

def cross_entropy_loss(mask, output):
    # The shape of the mask is (B, 3, 224, 224), the shape of the output is (B, 3, 224, 224)

    # Because the CrossEntropyLoss already includes the softmax function, the output does not need to be processed by the softmax function

    # Set the loss function, because the mask is not binary, dog is 1, cat is 2, background is 0, so use CrossEntropyLoss
    # Convert the one-hot mask to a class index mask
    mask = mask.argmax(dim=1)
    return nn.CrossEntropyLoss()(output, mask)

def dice_loss(mask, output, smooth=1e-5):
    """
    Computes the Dice loss for multi-class segmentation.

    Args:
        mask: Tensor of shape (B, C, H, W), one-hot encoded ground truth.
        output: Tensor of shape (B, C, H, W), raw logits or softmax probabilities.
        smooth: Smoothing factor to prevent division by zero.

    Returns:
        Average Dice loss over all classes.
    """
    # Ensure output is in probability space (apply softmax if needed)
    output = torch.softmax(output, dim=1)  # Apply softmax for probability distribution

    # Flatten the masks and outputs
    mask_flat = mask.contiguous().view(mask.shape[0], mask.shape[1], -1)
    output_flat = output.contiguous().view(output.shape[0], output.shape[1], -1)

    # Compute intersection and union
    intersection = (mask_flat * output_flat).sum(dim=2)
    union = mask_flat.sum(dim=2) + output_flat.sum(dim=2)

    # Compute Dice coefficient
    dice_score = (2. * intersection + smooth) / (union + smooth)

    # Compute loss (1 - Dice score), and average over classes
    dice_loss_value = 1 - dice_score.mean()

    return dice_loss_value


def iou_loss(mask, output, smooth=1e-5):
    """
    Computes the IoU loss for multi-class segmentation.

    Args:
        mask: Tensor of shape (B, C, H, W), one-hot encoded ground truth.
        output: Tensor of shape (B, C, H, W), raw logits or softmax probabilities.
        smooth: Smoothing factor to prevent division by zero.

    Returns:
        Average IoU loss over all classes.
    """
    # Ensure output is in probability space (apply softmax if needed)
    output = torch.softmax(output, dim=1)

    # Flatten the masks and outputs
    mask_flat = mask.contiguous().view(mask.shape[0], mask.shape[1], -1)
    output_flat = output.contiguous().view(output.shape[0], output.shape[1], -1)

    # Compute intersection and union
    intersection = (mask_flat * output_flat).sum(dim=2)
    union = mask_flat.sum(dim=2) + output_flat.sum(dim=2) - intersection

    # Compute IoU score
    iou_score = (intersection + smooth) / (union + smooth)

    # Compute loss (1 - IoU score), and average over classes
    iou_loss_value = 1 - iou_score.mean()

    return iou_loss_value


def total_seg_loss(mask, output):
    # The shape of the mask is (B, 3, 224, 224), the shape of the output is (B, 3, 224, 224)
    alpha_value = 1
    beta_value = 1
    gamma_value = 1
    return alpha_value * cross_entropy_loss(mask, output) + beta_value * dice_loss(mask, output) + gamma_value * iou_loss(mask, output)

def train():

    # SETTINGS
    project_name = 'CV-CLIP-Seg-Vit_L_14'  # Set the project name
    num_epochs = 1000
    lr = 1e-4
    weight_decay = 1e-5
    train_batch_size = 64
    test_batch_size = 64
    patience = 100
    num_workers = 16 # Number of CPU processes for data preprocessing, the maximum value of the server is 100
    log_path = './logs/CLIP_Seg_Vit_L_14/'

    # Set the environment variable
    os.environ["WANDB_API_KEY"] = "65e89d6040ee39f44b12f957c13c2af040aed83e"

    # Set the device
    torch.cuda.init()
    device_ids = [0, 1]  # Use multiple GPUs, the maximum GPU number of the server is 8 H100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the dataset
    train_dataset = SegDataset(root_dir="new_dataset", split="train", transform=True) # Apply augmentations
    test_dataset = SegDataset(root_dir="new_dataset", split="test", transform=None)  # No augmentation for test set
    train_loader  = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              prefetch_factor=4, persistent_workers=True
                              )
    test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            prefetch_factor=4, persistent_workers=True
                            )

    # Load the model

    # pretrained_dict = "logs/CLIP/best_model.pt"
    # clip_model = torch.load(pretrained_dict,weights_only=False)
    #
    # # If the model is saved as DataParallel, remove the DataParallel module
    # if isinstance(clip_model, torch.nn.DataParallel):
    #     clip_model = clip_model.module

    # Load a pre-trained CLIP model (e.g. ViT-B/32)
    # 512
    # clip_model, preprocess = clip.load("ViT-B/16", device=device, download_root="/mnt/ceph_rbd/users/jingyu/.cache/clip")
    # clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="/mnt/ceph_rbd/users/jingyu/.cache/clip")

    # 768
    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root="/mnt/ceph_rbd/users/jingyu/.cache/clip")

    # Freeze the clip model
    for param in clip_model.parameters():
        param.requires_grad = False

    print("The number of parameters in the clip models: ", sum(p.numel() for p in clip_model.parameters()))
    print("The number of trainable parameters in the clip models: ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad))

    # -----------------------------------------------------------------

    # Define category prompts
    label_to_prompt = {
        "cat": "a photo of a cat",
        "dog": "a photo of a dog"
    }

    text_tokens = clip.tokenize(list(label_to_prompt.values())).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)
        text_features = text_features.detach()

    model = LSegNet(clip_model, text_features, num_classes=3, features=768)

    model = nn.DataParallel(model, device_ids=device_ids)  # Set the model to use multiple GPUs
    model.to(device)

    early_stop = EarlyStopping(log_path=log_path, patience=patience, verbose=True)

    # Set the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    # Set the wandb configuration
    current_step = 1
    wandb.init(
        # set the wandb project where this run will be logged
        project = project_name,
        # track hyperparameters and run metadata
        config = {
            "epochs": num_epochs,
            "batches_train": train_batch_size,
            "batches_test": test_batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
        },
        # Current date and time is set as the name of this run
        name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # ------- Print information
    print("--------------------------------------------------------------")
    print("The device is: ", device)
    print("The number of training data is: ", len(train_dataset))
    print("The number of testing data is: ", len(test_dataset))
    print("The number of parameters in the models: ", sum(p.numel() for p in model.parameters()))
    print("The number of trainable parameters in the models: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("The batch size of training data is: ", train_batch_size)
    print("The batch size of testing data is: ", test_batch_size)
    print("The learning rate is: ", lr)
    print("The weight decay is: ", weight_decay)
    print("The number of epochs is: ", num_epochs)
    print("The patience is: ", patience)
    print("--------------------------------------------------------------")

    # Color definition
    RED_COLOR   = "\033[31m"
    RESET_COLOR = "\033[0m"

    start_time = time.time()
    train_epoch_loss = []
    test_epoch_loss = []

    # Train and test the models
    for epoch in range(num_epochs):

        # ================================================================================
        #                                   Training
        # ================================================================================
        # Set the total loss
        total_loss = 0.0

        # Set the progress bar
        pbar = tqdm(train_loader, total=len(train_loader), ncols=160, colour="red", file=sys.stdout)

        # Set the models to train mode
        model.train()
        for i, (image, mask, text_description) in enumerate(pbar):

            # Set the image and mask to device
            image = image.to(device)
            mask = mask.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            seg_logits = model(image)  # Forward pass

            loss = total_seg_loss(mask, seg_logits)  # Compute the loss
            loss.backward()                    # Backward pass to compute accumulated gradients
            optimizer.step()                   # Perform parameter update based on current gradients

            # Add the mini-batch training loss to epoch loss
            total_loss += loss.item()

            # ------- Update the progress bar
            tqdm_epoch = f"{'Epoch:':<6}{f'{epoch+1:03d}/{num_epochs}':>8}{' | Train ':<8}"
            tqdm_lr = f"{'| LR:':<5}{f'{lr:12.8f}':>15}{' '}"
            tqdm_loss = f"{'| Loss:':<5}{f'{total_loss / (i + 1):12.8f}':>15}{' '}"
            tqdm_time = f"{'| Time:':<5}{f'{(time.time() - start_time) / 3600.0:8.4f}':>10}{' hours'}"
            s = f"{RED_COLOR}{tqdm_epoch + tqdm_lr + tqdm_loss + tqdm_time}{RESET_COLOR}"
            pbar.set_description(s)

        # Log the loss
        train_epoch_loss.append(total_loss / len(train_loader))
        wandb.log({"train_loss": total_loss / len(train_loader)}, step=current_step)

        # ================================================================================
        #                                  Testing
        # ================================================================================
        # Set the total loss
        total_loss = 0.0

        # Set the progress bar
        pbar = tqdm(test_loader, total=len(test_loader), ncols=160, colour="yellow", file=sys.stdout)

        # Set the models to evaluation
        model.eval()
        with torch.no_grad():
            for i, (image, mask, text_description) in enumerate(pbar):

                # Set the image and mask to device
                image = image.to(device)
                mask = mask.to(device)

                seg_logits = model(image)  # Forward pass

                loss = total_seg_loss(mask, seg_logits)  # Compute the loss

                # Add the mini-batch training loss to epoch loss
                total_loss += loss.item()

                # # ------- Update the progress bar
                tqdm_epoch = f"{'':<14}{'   Valid ':<8}"
                tqdm_lr = f"{'| LR:':<5}{f'{lr:12.8f}':>15}{' '}"
                tqdm_loss = f"{'| Loss:':<5}{f'{total_loss / (i + 1):12.8f}':>15}{' '}"
                tqdm_time = f"{'| Time:':<5}{f'{(time.time() - start_time) / 3600.0:8.4f}':>10}{' hours'}"
                s = f"{RED_COLOR}{tqdm_epoch + tqdm_lr + tqdm_loss + tqdm_time}{RESET_COLOR}"
                pbar.set_description(s)

        # Log the loss
        test_epoch_loss.append(total_loss / len(test_loader))
        wandb.log({"valid_loss": total_loss / len(test_loader)}, step=current_step)
        current_step += 1

        # NOTE: Save models parameters when test loss decreases. If test
        #  loss doesn't decrease after a given patience, early stops the training.
        early_stop(total_loss / len(test_loader), model)
        if early_stop.early_stop:
            print(f"\n{RED_COLOR}Early stopping happened at No.{epoch+1} epoch.\n{RESET_COLOR}")
            break

        # # Save models parameters at a given interval
        # if (epoch+1) % save_interval == 0:
        #     print(f"{BLUE_COLOR}Saving parameters at No.{epoch + 1} epoch...{RESET_COLOR}")
        #     torch.save(model.state_dict(), log_path + '/epoch' + str(epoch+1) + '_param.pth')

    # Finish the wandb
    wandb.finish()

    # Save the best models
    model.load_state_dict(torch.load(early_stop.log_path + '/best_param.pth', weights_only=True))  # Load the best parameters
    torch.save(model, log_path + '/best_model.pt')


if __name__ == "__main__":
    train()

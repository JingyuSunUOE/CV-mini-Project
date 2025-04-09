# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.SAM.point_prompt import PointPromptEncoder
from models.SAM.sam_encoder import ImageEncoderViT
from models.SAM.sam_decoder import MaskDecoder
from models.SAM.transformer import TwoWayTransformer
from models.SAM.point_prompt import generate_point_prompt_and_binary_mask
from models.SAM.sam import SAM
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

def binary_cross_entropy_loss(mask, output):
    """
    使用 BCEWithLogitsLoss 计算二分类分割损失
    输入:
        mask:    (B, 1, H, W)，值为 0 或 1
        output:  (B, 1, H, W)，未过 sigmoid 的 raw logits
    """
    return nn.BCEWithLogitsLoss()(output, mask.float())


def binary_dice_loss(mask, output, smooth=1e-5):
    """
    Dice loss for binary segmentation.
    输入:
        mask: (B, 1, H, W), ground truth binary mask
        output: (B, 1, H, W), raw logits
    """
    probs = torch.sigmoid(output)
    mask_flat = mask.view(mask.size(0), -1)
    probs_flat = probs.view(probs.size(0), -1)

    intersection = (mask_flat * probs_flat).sum(dim=1)
    union = mask_flat.sum(dim=1) + probs_flat.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def binary_iou_loss(mask, output, smooth=1e-5):
    """
    IoU loss for binary segmentation.
    输入:
        mask: (B, 1, H, W)
        output: (B, 1, H, W)
    """
    probs = torch.sigmoid(output)
    mask_flat = mask.view(mask.size(0), -1)
    probs_flat = probs.view(probs.size(0), -1)

    intersection = (mask_flat * probs_flat).sum(dim=1)
    union = mask_flat.sum(dim=1) + probs_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()


def total_binary_seg_loss(mask, output, alpha=1.0, beta=1.0, gamma=1.0):
    """
    综合 BCE + Dice + IoU
    输入:
        mask: (B, 1, H, W)，值为0或1
        output: (B, 1, H, W)，raw logits
    """
    return (alpha * binary_cross_entropy_loss(mask, output) +
            beta * binary_dice_loss(mask, output) +
            gamma * binary_iou_loss(mask, output))


def train():

    # SETTINGS
    project_name = 'CV-SAM'  # Set the project name
    num_epochs = 2000
    lr = 1e-4
    weight_decay = 1e-5
    train_batch_size = 64
    test_batch_size = 64
    patience = 200
    num_workers = 16 # Number of CPU processes for data preprocessing, the maximum value of the server is 100
    log_path = './logs/SAM/'

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

    decoder = MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,))
    point_encoder = PointPromptEncoder(embed_dim=256, input_image_size=(224, 224))
    image_encoder = ImageEncoderViT(img_size=224, depth=4, num_heads=4, global_attn_indexes=[2, 5, 8, 11])

    # -----------------------------------------------------------------
    model = SAM(image_encoder, point_encoder, decoder)

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

            # Get randomly sampled points, labels, and updated masks through generate_dynamic_point_prompt_and_update_mask
            coords_tensor, labels_tensor, binary_masks, selected_classes = generate_point_prompt_and_binary_mask(mask)

            coords_tensor = coords_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            binary_masks = binary_masks.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            predicted_masks = model(image, coords_tensor, labels_tensor)  # Forward pass

            loss = total_binary_seg_loss(mask=binary_masks, output=predicted_masks)  # Compute the loss
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

                # Get randomly sampled points, labels, and updated masks through generate_dynamic_point_prompt_and_update_mask
                coords_tensor, labels_tensor, binary_masks, selected_classes = generate_point_prompt_and_binary_mask(mask)

                coords_tensor = coords_tensor.to(device)
                labels_tensor = labels_tensor.to(device)
                binary_masks = binary_masks.to(device)

                predicted_masks = model(image, coords_tensor, labels_tensor)  # Forward pass

                loss = total_binary_seg_loss(mask=binary_masks, output=predicted_masks)  # Compute the loss

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

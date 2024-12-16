import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_dataloader
from unet_model import UNet, ResidualBlock
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models

def sobel_edge_loss(predictions, targets):
    """Compute edge loss using Sobel filters for multi-channel inputs."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=predictions.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    # Convert multi-channel input to single-channel by averaging across channels
    predictions_gray = predictions.mean(dim=1, keepdim=True)
    targets_gray = targets.mean(dim=1, keepdim=True)

    # Apply Sobel filters
    pred_edge_x = nn.functional.conv2d(predictions_gray, sobel_x, padding=1)
    pred_edge_y = nn.functional.conv2d(predictions_gray, sobel_y, padding=1)
    target_edge_x = nn.functional.conv2d(targets_gray, sobel_x, padding=1)
    target_edge_y = nn.functional.conv2d(targets_gray, sobel_y, padding=1)

    # Compute edge magnitude
    pred_edges = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
    target_edges = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

    # Return L1 loss between predicted and target edges
    return nn.functional.l1_loss(pred_edges, target_edges)

def train_model(
    input_dir,
    output_dir,
    batch_size=1,
    epochs=100,
    learning_rate=1e-4,
    checkpoint_path="./checkpoints/final_model.pth",
    accumulation_steps=4
):
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    if local_rank == 0:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader = create_dataloader(
        input_dir,
        output_dir,
        batch_size=batch_size,
        shuffle=(not distributed),
        num_workers=4,
        distributed=distributed
    )

    model = UNet()
    model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Loss functions
    criterion = nn.L1Loss()
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # Warm Restarts scheduling

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        if distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss_l1 = criterion(outputs, targets) / accumulation_steps

                # Perceptual Loss
                features_outputs = vgg(outputs)
                features_targets = vgg(targets)
                loss_perceptual = nn.functional.l1_loss(features_outputs, features_targets) / accumulation_steps

                # Edge Loss
                loss_edge = sobel_edge_loss(outputs, targets) / accumulation_steps

                # Total Loss
                loss = loss_l1 + 0.05 * loss_perceptual + 0.1 * loss_edge

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

        scheduler.step(epoch + batch_idx / len(dataloader))

        # Adaptive Batch Size Increase
        if epoch > epochs * 0.7:  # Increase batch size after 70% of training
            batch_size = min(batch_size + 1, 4)
            dataloader = create_dataloader(
                input_dir,
                output_dir,
                batch_size=batch_size,
                shuffle=(not distributed),
                num_workers=4,
                distributed=distributed
            )

        if local_rank == 0:
            epoch_loss /= len(dataloader)
            if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == epochs - 1):
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    if local_rank == 0:
        if distributed:
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        print(f"Final model saved to {checkpoint_path}.")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    input_dir = "./datasets/train/preprocessed_input"
    output_dir = "./datasets/train/preprocessed_output"
    train_model(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=1,
        epochs=100,
        learning_rate=1e-4,
        checkpoint_path="./checkpoints/final_model.pth",
        accumulation_steps=4
    )

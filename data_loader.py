import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
import torch

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, img_size=(512, 512), normalize=True):
        """
        Custom dataset for loading input-output image pairs.
        Args:
            input_dir (str): Path to input image directory.
            output_dir (str): Path to output image directory.
            img_size (tuple): Target size for resizing images.
            normalize (bool): Normalize pixel values to [0, 1] if True.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.normalize = normalize
        self.input_files = sorted(os.listdir(input_dir))
        self.output_files = sorted(os.listdir(output_dir))

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        Retrieves an input-output image pair at the specified index.
        Args:
            idx (int): Index of the image pair.
        Returns:
            tuple: Preprocessed input and output images as tensors.
        """
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])

        # Load and preprocess input image
        input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, self.img_size)
        if self.normalize:
            input_img = input_img / 255.0

        # Load and preprocess output image
        output_img = cv2.imread(output_path, cv2.IMREAD_COLOR)
        output_img = cv2.resize(output_img, self.img_size)
        if self.normalize:
            output_img = output_img / 255.0

        # Convert images to tensors
        input_tensor = torch.tensor(input_img.transpose(2, 0, 1), dtype=torch.float32)
        output_tensor = torch.tensor(output_img.transpose(2, 0, 1), dtype=torch.float32)

        return input_tensor, output_tensor


def create_dataloader(input_dir, output_dir, batch_size=16, shuffle=True, num_workers=2, distributed=False):
    """
    Creates a DataLoader for input-output image pairs.
    Args:
        input_dir (str): Path to input image directory.
        output_dir (str): Path to output image directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Shuffle the dataset if True.
        num_workers (int): Number of subprocesses to use for data loading.
        distributed (bool): Whether to use DistributedSampler for distributed training.
    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    dataset = ImageDataset(input_dir, output_dir)
    
    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True 
    )
    return dataloader


if __name__ == "__main__":
    # Example usage
    input_dir = "./datasets/train/input"
    output_dir = "./datasets/train/output"
    dataloader = create_dataloader(input_dir, output_dir, distributed=False)

    for batch_idx, (inputs, outputs) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Input batch shape: {inputs.shape}")
        print(f"Output batch shape: {outputs.shape}")
        break

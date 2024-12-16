import os
import cv2
import torch
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from unet_model import UNet
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

class InferenceDataset(Dataset):
    def __init__(self, input_dir, ground_truth_dir, img_size=(512, 512), normalize=True):
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.img_size = img_size
        self.normalize = normalize
        self.input_files = sorted(os.listdir(input_dir))
        self.ground_truth_files = sorted(os.listdir(ground_truth_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        ground_truth_file = self.ground_truth_files[idx]

        input_path = os.path.join(self.input_dir, input_file)
        ground_truth_path = os.path.join(self.ground_truth_dir, ground_truth_file)

        input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, self.img_size)
        if self.normalize:
            input_img = input_img / 255.0
        input_tensor = torch.tensor(input_img.transpose(2, 0, 1), dtype=torch.float32)

        ground_truth_img = cv2.imread(ground_truth_path, cv2.IMREAD_COLOR)
        ground_truth_img = cv2.resize(ground_truth_img, self.img_size)
        if self.normalize:
            ground_truth_img = ground_truth_img / 255.0
        ground_truth_tensor = torch.tensor(ground_truth_img.transpose(2, 0, 1), dtype=torch.float32)

        return input_tensor, ground_truth_tensor, input_file

def infer(
    model_path,
    input_dir,
    output_dir,
    ground_truth_dir,
    img_size=(512, 512),
    win_size=11,
    batch_size=1
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

    model = UNet()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset = InferenceDataset(input_dir, ground_truth_dir, img_size=img_size)
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if local_rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    progress_bar = tqdm(dataloader, desc="Running Inference", disable=(local_rank != 0))
    with torch.no_grad():
        for inputs, ground_truths, file_names in progress_bar:
            inputs = inputs.to(device)
            ground_truths = ground_truths.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)

            if local_rank == 0:
                for i in range(outputs.size(0)):
                    output_tensor = outputs[i]
                    input_tensor = inputs[i]
                    ground_truth_tensor = ground_truths[i]
                    file_name = file_names[i]

                    image_folder = os.path.join(output_dir, f"image_{num_images}")
                    os.makedirs(image_folder, exist_ok=True)

                    restored_path = os.path.join(image_folder, f"restored_{num_images}.png")
                    input_path = os.path.join(image_folder, f"input_{num_images}.png")
                    ground_truth_path = os.path.join(image_folder, f"ground_truth_{num_images}.png")
                    edge_path = os.path.join(image_folder, f"edge_{num_images}.png")

                    save_image(output_tensor, restored_path)
                    save_image(input_tensor, input_path)
                    save_image(ground_truth_tensor, ground_truth_path)
                    save_edge_image(output_tensor, edge_path)  

                    num_images += 1

            outputs_cpu = outputs.cpu()
            ground_truths_cpu = ground_truths.cpu()
            for i in range(outputs.size(0)):
                output_img = outputs_cpu[i].permute(1, 2, 0).numpy()
                ground_truth_img = ground_truths_cpu[i].permute(1, 2, 0).numpy()

                min_dim = min(ground_truth_img.shape[:2])
                adjusted_win_size = win_size
                if min_dim < win_size:
                    adjusted_win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                    if adjusted_win_size < 3:
                        adjusted_win_size = 3

                image_ssim = ssim(
                    ground_truth_img,
                    output_img,
                    data_range=1.0,
                    win_size=adjusted_win_size,
                    channel_axis=-1
                )

                image_psnr = psnr(ground_truth_img, output_img, data_range=1.0)

                total_psnr += image_psnr
                total_ssim += image_ssim

    if distributed:
        total_psnr_tensor = torch.tensor(total_psnr, device=device)
        total_ssim_tensor = torch.tensor(total_ssim, device=device)
        num_images_tensor = torch.tensor(num_images, device=device)

        dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_images_tensor, op=dist.ReduceOp.SUM)

        total_psnr = total_psnr_tensor.item()
        total_ssim = total_ssim_tensor.item()
        num_images = num_images_tensor.item()

    if local_rank == 0 and num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    elif local_rank == 0:
        print("No images were valid for SSIM calculation.")

    if distributed:
        dist.destroy_process_group()

def save_image(image_tensor, output_path):
    """
    텐서를 이미지 파일로 저장합니다.
    Args:
        image_tensor (torch.Tensor): 저장할 출력 이미지 텐서
        output_path (str): 출력 이미지 저장 경로
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    cv2.imwrite(output_path, img)

def save_edge_image(image_tensor, output_path):
    """
    모델 출력 이미지의 엣지 정보를 Sobel 필터를 사용해 계산하고 저장합니다.
    Args:
        image_tensor (torch.Tensor): 출력 이미지 텐서
        output_path (str): 저장 경로
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    img = (img * 255).clip(0, 255).astype("uint8")  

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")
    sobel_y = sobel_x.T
    edge_x = cv2.filter2D(img, cv2.CV_32F, sobel_x)  
    edge_y = cv2.filter2D(img, cv2.CV_32F, sobel_y)

    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
    edge = (edge / edge.max() * 255).clip(0, 255).astype("uint8")  

    cv2.imwrite(output_path, edge)

if __name__ == "__main__":
    model_path = "./checkpoints/final_model.pth"
    input_dir = "./datasets/test/preprocessed_input"
    output_dir = "./output"
    ground_truth_dir = "./datasets/test/ground_truth"

    infer(
        model_path,
        input_dir,
        output_dir,
        ground_truth_dir,
        batch_size=1
    )

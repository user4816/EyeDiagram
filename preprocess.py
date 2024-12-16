import os
import cv2
import numpy as np
from tqdm import tqdm

class Preprocessor:
    def __init__(self, input_dir, output_dir, save_input_dir, save_output_dir, img_size=(512, 512), normalize=True):
        """
        Initializes the Preprocessor with directories and options.
        Args:
            input_dir (str): Path to the directory containing input images.
            output_dir (str): Path to the directory containing output images.
            save_input_dir (str): Path to save preprocessed input images.
            save_output_dir (str): Path to save preprocessed output images.
            img_size (tuple): Target size for resizing images (width, height).
            normalize (bool): Whether to normalize pixel values to [0, 1].
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_input_dir = save_input_dir
        self.save_output_dir = save_output_dir
        self.img_size = img_size
        self.normalize = normalize

        # Create directories if they don't exist
        os.makedirs(self.save_input_dir, exist_ok=True)
        os.makedirs(self.save_output_dir, exist_ok=True)

    def load_image(self, path):
        """
        Loads an image from the specified path.
        Args:
            path (str): Path to the image file.
        Returns:
            np.ndarray: The loaded image.
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image at {path} could not be loaded.")
        return img

    def preprocess_image(self, img):
        """
        Applies preprocessing to an image.
        Args:
            img (np.ndarray): Input image to preprocess.
        Returns:
            np.ndarray: Preprocessed image.
        """
        img = cv2.resize(img, self.img_size)
        if self.normalize:
            img = img / 255.0
        return img

    def save_image(self, img, path):
        """
        Saves a preprocessed image to the specified path.
        Args:
            img (np.ndarray): Image to save.
            path (str): Path to save the image.
        """
        img = (img * 255).clip(0, 255).astype("uint8")  # Scale back to [0, 255]
        cv2.imwrite(path, img)

    def preprocess_and_save(self):
        """
        Preprocesses the dataset and saves preprocessed images.
        """
        input_files = sorted(os.listdir(self.input_dir))
        output_files = sorted(os.listdir(self.output_dir))

        for input_file, output_file in tqdm(zip(input_files, output_files), total=len(input_files), desc="Processing Images"):
            input_path = os.path.join(self.input_dir, input_file)
            output_path = os.path.join(self.output_dir, output_file)

            save_input_path = os.path.join(self.save_input_dir, input_file)
            save_output_path = os.path.join(self.save_output_dir, output_file)

            # Load and preprocess images
            input_img = self.preprocess_image(self.load_image(input_path))
            output_img = self.preprocess_image(self.load_image(output_path))

            # Save preprocessed images
            self.save_image(input_img, save_input_path)
            self.save_image(output_img, save_output_path)


if __name__ == "__main__":
    # Example usage for train data
    train_input_dir = "./datasets/train/input"
    train_output_dir = "./datasets/train/output"
    train_preprocessed_input_dir = "./datasets/train/preprocessed_input"
    train_preprocessed_output_dir = "./datasets/train/preprocessed_output"

    train_preprocessor = Preprocessor(
        input_dir=train_input_dir,
        output_dir=train_output_dir,
        save_input_dir=train_preprocessed_input_dir,
        save_output_dir=train_preprocessed_output_dir
    )
    train_preprocessor.preprocess_and_save()

    # Example usage for test data
    test_input_dir = "./datasets/test/input"
    test_output_dir = "./datasets/test/ground_truth"
    test_preprocessed_input_dir = "./datasets/test/preprocessed_input"
    test_preprocessed_output_dir = "./datasets/test/preprocessed_ground_truth"

    test_preprocessor = Preprocessor(
        input_dir=test_input_dir,
        output_dir=test_output_dir,
        save_input_dir=test_preprocessed_input_dir,
        save_output_dir=test_preprocessed_output_dir
    )
    test_preprocessor.preprocess_and_save()

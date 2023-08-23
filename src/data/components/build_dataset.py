import os
import tarfile
import zipfile

import autorootcwd  # noqa: F401
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.image_encoder import ImageEncoder
from rich.progress import Progress

class DataParser:
    def __init__(self, expected_label_length=5):
        """Initialize the data parser with the expected length of the label."""
        self.expected_label_length = expected_label_length

    def __process_image(self, file, file_name):
        """Process a single image by normalizing and resizing.

        Args:
            file: File object of the image.
            file_name: Name of the image file.

        Returns:
            Tuple of processed image and corresponding label.
        """
        image = Image.open(file).convert('L')
        label = os.path.splitext(os.path.basename(file_name))[0]
        if len(label) != self.expected_label_length:
            return None, None
        if image.size != (58, 20):
            image = image.resize((58, 20))
        image_array = np.array(image)
        image_normalized = image_array / 255.0
        return image_normalized, label

    def process_images(self, file_path, save_path):
        """Process images from a given path and save them.

        Args:
            file_path: Path to the directory or ZIP file containing images.
            save_path: Path to save the processed images and labels.

        Returns:
            Tuple of lists containing processed images and labels.
        """
        processed_images = []
        labels = []
        if file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zipf:
                for file_name in zipf.namelist():
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):
                        with zipf.open(file_name) as file:
                            image, label = self.__process_image(file, file_name)
                            if image is not None:
                                processed_images.append(image)
                                labels.append(label)
        elif file_path.endswith("tar.gz"):
            with tarfile.open(file_path, 'r:gz') as tarf:
                for file_name in tarf.getnames():
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):
                        with tarf.extractfile(file_name) as file:
                            image, label = self.__process_image(file, file_name)
                            if image is not None:
                                processed_images.append(image)
                                labels.append(label)
        else:
            image_paths = [f for f in os.listdir(file_path) if f.endswith('.png') or f.endswith('.jpg')]
            for image_path in image_paths:
                with open(f"{file_path}/{image_path}", 'rb') as file:
                    image, label = self.__process_image(file, f"{file_path}/{image_path}")
                    if image is not None:
                        processed_images.append(image)
                        labels.append(label)

        np.savez(save_path, images=processed_images, labels=labels)
        return processed_images, labels


class CaptchaDataset(Dataset):
    def __init__(self, npz_file):
        """Initialize the dataset with the given file path.

        Args:
            npz_file: Path to the NPZ file containing images and labels.
        """
        loaded_data = np.load(npz_file)
        self.images = loaded_data['images']
        self.labels = loaded_data['labels']

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Return the data at the specified index.

        Args:
            index: Index of the data to retrieve.

        Returns:
            Tuple containing image tensor and encoded label.
        """
        image = self.images[index]
        label = self.labels[index]
        encoded_label = ImageEncoder().encode_labels([label])[0]  # 使用你的編碼函數
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(-1)  # 添加一個通道維度
        return image_tensor.permute(2, 0, 1), encoded_label  # 改變維度順序以符合PyTorch的期望

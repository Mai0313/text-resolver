import os
import zipfile
import glob

import autorootcwd  # noqa: F401
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.image_encoder import ImageEncoder

class DataParser:
    def __init__(self, expected_label_length=5):
        self.expected_label_length = expected_label_length

    def __process_image(self, file, file_name):
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
        processed_images = []
        labels = []
        if file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zipf:
                with tqdm(zipf.namelist(), desc="Processing ZIP") as pbar:
                    for file_name in pbar:
                        if file_name.endswith('.png') or file_name.endswith('.jpg'):
                            with zipf.open(file_name) as file:
                                image, label = self.__process_image(file, file_name)
                                if image is not None:
                                    processed_images.append(image)
                                    labels.append(label)
                                    pbar.set_postfix({"Current label": label})
        else:
            image_paths = [f for f in os.listdir(file_path) if f.endswith('.png') or f.endswith('.jpg')]
            with tqdm(image_paths, desc="Processing images") as pbar:
                for image_path in pbar:
                    with open(f"{file_path}/{image_path}", 'rb') as file:
                        image, label = self.__process_image(file, f"{file_path}/{image_path}")
                        if image is not None:
                            processed_images.append(image)
                            labels.append(label)
                            pbar.set_postfix({"Current label": label})

        np.savez(save_path, images=processed_images, labels=labels)
        return processed_images, labels

class CaptchaDataset(Dataset):
    def __init__(self, npz_file):
        loaded_data = np.load(npz_file)
        self.images = loaded_data['images']
        self.labels = loaded_data['labels']

    def __len__(self):
        """返回數據集的大小."""
        return len(self.images)

    def __getitem__(self, index):
        """返回索引處的數據。"""
        image = self.images[index]
        label = self.labels[index]
        encoded_label = ImageEncoder().encode_labels([label])[0]  # 使用你的編碼函數
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(-1)  # 添加一個通道維度
        return image_tensor.permute(2, 0, 1), encoded_label  # 改變維度順序以符合PyTorch的期望

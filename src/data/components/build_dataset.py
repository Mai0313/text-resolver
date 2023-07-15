import os
import zipfile

import autorootcwd  # noqa: F401
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.image_encoder import ImageEncoder

class DataPaser:
    def __init__(self, expected_label_length=5):
        self.expected_label_length = expected_label_length

    def process_images(self, zip_file_path, save_path):
        processed_images = []
        labels = []
        with zipfile.ZipFile(zip_file_path, 'r') as zipf:
            for file_name in zipf.namelist():
                if file_name.endswith('.png'):
                    with zipf.open(file_name) as file:
                        image = Image.open(file).convert('L')
                        label = os.path.splitext(os.path.basename(file_name))[0]
                        if len(label) != self.expected_label_length:
                            continue
                        if image.size != (58, 20):
                            image = image.resize((58, 20))
                        image_array = np.array(image)
                        image_normalized = image_array / 255.0
                        processed_images.append(image_normalized)
                        labels.append(label)

        np.savez(save_path, images=processed_images, labels=labels)
        return processed_images, labels
    
    def process_test_images(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('L')
            if img.size != (20, 58):
                img = img.resize((20, 58))
            img_array = np.array(img)
            img_normalized = img_array / 255.0
        return img_normalized

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

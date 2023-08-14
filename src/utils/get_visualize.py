import autorootcwd  # noqa: F401
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.image_encoder import ImageEncoder

class DataVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def visualize_prediction(self, images, labels):
        indices = np.random.choice(len(images), 10, replace=False)  # 隨機選擇10個索引

        fig, axs = plt.subplots(5, 2, figsize=(10, 25))
        for i, index in enumerate(indices):
            test_image = images[index].cpu().numpy().squeeze()  # 轉換為numpy array
            test_label = ImageEncoder().decode_output(labels[index])

            test_image_tensor = torch.tensor(test_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(test_image_tensor)
                pred_label = ImageEncoder().decode_output(output.squeeze())

            row = i // 2
            col = i % 2
            axs[row, col].imshow(test_image, cmap='gray')
            axs[row, col].set_title(f'Predicted: {pred_label}\nTrue: {test_label}')
            axs[row, col].axis('off')

        return fig

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
import matplotlib.pyplot as plt

from src.utils.image_encoder import ImageEncoder


class DataVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_accuracy(self, images, labels, num_classes):
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        correct_count = 0
        total_count = 0
        for i in range(len(images)):
            test_image = images[i].cpu().numpy().squeeze()
            test_label = ImageEncoder().decode_output(labels_one_hot[i])
            test_image_tensor = (
                torch.tensor(test_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                output = self.model(test_image_tensor)
                pred_label = ImageEncoder().decode_output(output.squeeze())
            if pred_label == test_label:
                correct_count += 1
            total_count += 1
        return correct_count, total_count

    def visualize_prediction(self, images, labels):
        """Visualize the prediction of the model."""
        labels_one_hot = F.one_hot(labels, num_classes=36).float()
        rng = np.random.default_rng()
        indices = rng.choice(len(images), 10, replace=False)

        correct_count = 0
        total_count = 0

        fig, axs = plt.subplots(5, 2, figsize=(10, 25))
        for i, index in enumerate(indices):
            test_image = images[index].cpu().numpy().squeeze()  # 轉換為numpy array
            test_label = ImageEncoder().decode_output(labels_one_hot[index])

            test_image_tensor = (
                torch.tensor(test_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                output = self.model(test_image_tensor)
                pred_label = ImageEncoder().decode_output(output.squeeze())
            if pred_label == test_label:
                correct_count += 1
            total_count += 1

            row = i // 2
            col = i % 2
            axs[row, col].imshow(test_image)
            # axs[row, col].imshow(test_image, cmap = 'gray')
            axs[row, col].set_title(f"Predicted: {pred_label}\nTrue: {test_label}")
            axs[row, col].axis("off")

        accuracy = correct_count / total_count * 100
        return fig, accuracy

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class CrossEntropyLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 36)  # 將其轉換為適合CrossEntropyLoss的形狀
        labels_encoded = labels_encoded.view(-1)  # 將標籤轉換為一維張量
        original_loss = self.criterion(logits, labels_encoded)
        return original_loss * self.weight


class MSELoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.MSELoss(reduction="none")

    def __call__(self, prediction, images, labels_encoded):
        labels_one_hot = F.one_hot(labels_encoded, num_classes=36).float()
        logits = prediction.view(-1, 5, 36)
        original_loss = self.criterion(logits, labels_one_hot)
        return original_loss.mean() * self.weight


class CorrectnessLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 5, 36)
        preds = torch.argmax(logits, dim=2)
        correctness = preds == labels_encoded
        correctness_loss = 1.0 - torch.mean(correctness.float())
        return correctness_loss * self.weight


class AccuracyLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 5, 36)
        preds = torch.argmax(logits, dim=2)
        correctness = preds == labels_encoded
        return torch.mean(correctness.float()) * self.weight


class CorrectnessRewardLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 5, 36)
        preds = torch.argmax(logits, dim=2)
        correctness = preds == labels_encoded
        correctness_reward_loss = torch.mean(correctness.float())
        return correctness_reward_loss * self.weight

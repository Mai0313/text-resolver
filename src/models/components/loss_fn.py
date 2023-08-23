import torch
import torch.nn.functional as F


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
        return original_loss.mean()

class CorrectnessLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 5, 36)
        preds = torch.argmax(logits, dim=2)
        correctness = torch.all(preds == labels_encoded, dim=1)
        correctness_loss = 1.0 - torch.mean(correctness.float())
        return correctness_loss

class CorrectnessRewardLoss:
    def __init__(self, tag, weight):
        super().__init__()
        self.tag = tag
        self.weight = weight

    def __call__(self, prediction, images, labels_encoded):
        logits = prediction.view(-1, 5, 36)
        preds = torch.argmax(logits, dim=2)
        correctness = torch.all(preds == labels_encoded, dim=1)
        correctness_reward_loss = torch.mean(correctness.float())
        return correctness_reward_loss

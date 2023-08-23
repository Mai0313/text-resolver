from typing import Any

import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.get_visualize import DataVisualizer


class CaptchaModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fns: list[torch.nn.Module],
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.loss_fns = loss_fns

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels_encoded = batch

        prediction = self.forward(images)

        losses = {}  # a dict of {loss_fn_name: loss_value}
        losses["total_loss"] = 0.0
        for loss_fn in self.loss_fns:
            losses[loss_fn.tag] = loss_fn(prediction=prediction, images=images, labels_encoded=labels_encoded)
            losses["total_loss"] += losses[loss_fn.tag] * loss_fn.weight
        return losses, prediction, images, labels_encoded

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses, prediction, images, labels_encoded = self.model_step(batch)
        self.train_loss(losses.get("total_loss"))
        self.log('train/original_loss', losses.get("original_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/correctness_loss', losses.get("correctness_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/correctness_reward_loss', losses.get("correctness_reward_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', losses.get("total_loss"), on_step=False, on_epoch=True, prog_bar=True)
        return losses.get("total_loss")

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses, prediction, images, labels_encoded = self.model_step(batch)
        self.val_loss(losses.get("total_loss"))
        self.log('val/original_loss', losses.get("original_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/correctness_loss', losses.get("correctness_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/correctness_reward_loss', losses.get("correctness_reward_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', losses.get("total_loss"), on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 100 == 0:
            labels_one_hot = F.one_hot(labels_encoded, num_classes=36).float()
            fig = DataVisualizer(self.net, self.device).visualize_prediction(images, labels_one_hot)
            self.logger.experiment.add_figure('Predicted Images', fig, self.global_step)

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses, prediction, images, labels_encoded = self.model_step(batch)
        self.test_loss(losses.get("total_loss"))
        self.log('test/original_loss', losses.get("original_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/correctness_loss', losses.get("correctness_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/correctness_reward_loss', losses.get("correctness_reward_loss"), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/total_loss', losses.get("total_loss"), on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CaptchaModule(None, None, None)

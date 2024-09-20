from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.get_visualize import DataVisualizer


class CaptchaModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fns: list[torch.nn.Module],
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.loss_fns = loss_fns

        # metric objects for calculating and averaging accuracy across batches
        self.num_classes = self.hparams.net.num_classes + 1
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # # Store these value for checking accuracy
        # self.correct_count = 0
        # self.total_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        current_batch_size = images.shape[0]

        prediction = self.forward(images)

        losses = {}  # a dict of {loss_fn_name: loss_value}
        losses["total_loss"] = 0.0
        for loss_fn in self.loss_fns:
            losses[loss_fn.tag] = loss_fn(
                prediction=prediction, images=images, labels_encoded=labels_encoded
            )
            losses["total_loss"] += losses[loss_fn.tag] * loss_fn.weight
        return losses, prediction, images, labels_encoded, current_batch_size

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses, prediction, _, labels_encoded, current_batch_size = self.model_step(batch)

        self.log_loss(losses, self.val_loss, current_batch_size, prefix="train")
        self.log_accuracy(
            prediction, labels_encoded, self.val_acc, current_batch_size, prefix="train"
        )
        return losses.get("total_loss")

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses, prediction, images, labels_encoded, current_batch_size = self.model_step(batch)

        self.log_loss(losses, self.val_loss, current_batch_size, prefix="val")
        self.log_accuracy(
            prediction, labels_encoded, self.val_acc, current_batch_size, prefix="val"
        )

        if batch_idx % 100 == 0:
            fig, accuracy = DataVisualizer(self.net, self.device).visualize_prediction(
                images, labels_encoded
            )
            self.logger.experiment.add_figure("Predicted_Images", fig, self.global_step)
            self.logger.experiment.add_scalar("Accuracy", accuracy, self.global_step)
            self.log("val/Accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses, prediction, _, labels_encoded, current_batch_size = self.model_step(batch)

        self.log_loss(losses, self.val_loss, current_batch_size, prefix="test")
        self.log_accuracy(
            prediction, labels_encoded, self.val_acc, current_batch_size, prefix="test"
        )

        # self.correct_count, self.total_count = DataVisualizer(self.net, self.device).get_accuracy(
        #     images, labels_encoded, self.num_classes
        # )

        # self.test_loss(losses.get("total_loss"))
        # for loss_name, loss_value in losses.items():
        #     self.log(f"test/{loss_name}", loss_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # accuracy = self.correct_count / self.total_count * 100
        # self.log("Test Dataset Accuracy", accuracy)
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    def log_loss(self, losses, loss_metrics, batch_size, prefix="train"):
        # losses: dict of per loss result
        # loss_metrics: dict of metrics from the torchmetric
        # prefix: str will be added to the error log
        for loss_name, loss_value in losses.items():
            loss_value = loss_value.item()
            self.log(
                f"{prefix}/{loss_name}",
                loss_metrics(loss_value),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def log_accuracy(self, preds, y, accuracy_metrics, batch_size, prefix="train"):
        preds_accuracy = preds.argmax(dim=-1)
        accuracy_metrics(preds_accuracy, y)
        self.log(
            f"{prefix}/accuracy",
            accuracy_metrics(preds_accuracy, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )


if __name__ == "__main__":
    _ = CaptchaModule(None, None, None)

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_msssim import ssim

from modules import model
from utils import board
from utils.utils import calc_fhd


class DLModel(LightningModule):
    def __init__(self, hparams, img_size, c_bits, denormalize):
        super().__init__()
        self.hparams.update(hparams)
        self.img_size = img_size
        self.challenge_bits = c_bits
        self.denormalize = denormalize

        self.log = {}

        self.l1_criterion = nn.L1Loss()
        self.bce_criterion = nn.BCEWithLogitsLoss()

        self.model = model.Model(hparams["c_weight"], hparams["ns"], c_bits)

        self.test_results = None
        self.pred_challenges = []
        self.pred_responses = []

    def loss_function(self, real_response, gen_response):
        real_den = self.denormalize(real_response)
        gen_den = self.denormalize(gen_response)
        l1_criterion = nn.L1Loss()

        ssim_loss = 1 - ssim(real_den, gen_den)
        l1_loss = l1_criterion(real_den, gen_den)

        return ssim_loss, l1_loss

    def on_train_epoch_start(self):
        self.grad_fig, self.grad_ax = board.create_grad_fig("Model")

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.model(challenge)

        ssim_loss, l1_loss = self.loss_function(real_response, gen_response)
        loss = ssim_loss + l1_loss

        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean()
        epoch = self.current_epoch
        self.logger.experiment.add_figure("Gradients", self.grad_fig, epoch)
        self.logger.experiment.add_scalar("Training Loss", loss, epoch)

    def validation_step(self, batch, batch_idx):
        fhd, challenge, gen_response = self.val_test_step(batch, batch_idx)
        return fhd

    def test_step(self, batch, batch_idx):
        fhd, challenge, gen_response = self.val_test_step(batch, batch_idx)
        self.pred_challenges.append(challenge.cpu().numpy().astype(np.uint8))
        self.pred_responses.append(gen_response.squeeze().cpu().numpy() * 255)
        return fhd

    def val_test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.model(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        fhd = [
            calc_fhd(rr.cpu().numpy(), gr.cpu().numpy()) for rr, gr in
            zip(real_response, gen_response)
        ]

        if batch_idx == 0:
            self.log["challenge"] = challenge
            self.log["real_response"] = real_response
            self.log["gen_response"] = gen_response

        return fhd, challenge, gen_response

    def validation_epoch_end(self, outputs):
        fhds = np.hstack(outputs)
        epoch = self.current_epoch

        self.logger.experiment.add_scalar(
            "Validation FHD", np.mean(fhds), epoch
        )

        output_fig = board.get_output_figure(self.log)
        self.logger.experiment.add_figure("Output", output_fig, epoch)

    def test_epoch_end(self, outputs):
        fhds = np.hstack(outputs)
        self.logger.experiment.add_scalar("Test FHD", np.mean(fhds),
                                          self.current_epoch)

        self.pred_challenges = np.vstack(self.pred_challenges)
        self.pred_responses = np.vstack(self.pred_responses)

    def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        is_step = self.trainer.global_step % 5 == 0
        if is_step:
            board.plot_grad_flow(self.model.named_parameters(),
                                 self.grad_ax)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            (self.hparams.beta1, self.hparams.beta2)
        )
        return optimizer

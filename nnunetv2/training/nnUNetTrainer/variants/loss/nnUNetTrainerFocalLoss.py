import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_500epochs, nnUNetTrainer_2000epochs, nnUNetTrainerAdamCosine_2000epochs, nnUNetTrainerNAdam_2000epochs, nnUNetTrainerRAdam_2000epochs
from nnunetv2.training.loss.compound_losses import DC_and_FC_loss, DC_and_CE_loss
from nnunetv2.training.loss.focal import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from torch.optim import AdamW, NAdam
import numpy as np

class nnUNetTrainerFocalLossCosine(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        loss = FocalLoss(ignore_index=0)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.9, nesterov=True)

        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=400,          # 첫 주기 길이(에폭 단위)
            T_mult=2,        # 주기 확장 계수 (T_mult=1: 고정 주기)
            eta_min=1e-6     # 최소 학습률
        )
        return optimizer, lr_scheduler

class nnUNetTrainerFocalLossNAdamCosine(nnUNetTrainerNAdam_2000epochs):
    def _build_loss(self):
        loss = FocalLoss(ignore_index=0)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_optimizers(self):
        optimizer = NAdam(self.network.parameters(),
                        #   lr=self.initial_lr,
                          lr = 1e-2,
                          weight_decay=self.weight_decay)

        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=400,          # 첫 주기 길이(에폭 단위)
            T_mult=2,        # 주기 확장 계수 (T_mult=1: 고정 주기)
            eta_min=1e-6     # 최소 학습률
        )
        return optimizer, lr_scheduler
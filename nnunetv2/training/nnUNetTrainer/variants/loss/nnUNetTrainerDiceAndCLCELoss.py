import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_500epochs, nnUNetTrainer_2000epochs, nnUNetTrainerAdamCosine_2000epochs, nnUNetTrainerNAdam_2000epochs, nnUNetTrainerRAdam_2000epochs
from nnunetv2.hs_custom.clce_loss_cesaracebes import dice_clCE_loss
from nnunetv2.training.loss.focal import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmUpRestarts
import numpy as np

class nnUNetTrainerDiceAndCLCELoss(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = dice_clCE_loss()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerDiceAndCLCELossCosine(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = dice_clCE_loss()
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), 1e-6, weight_decay=self.weight_decay,
                                    momentum=0.9, nesterov=True)
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=200,      
            T_mult=2,
            T_up=10,
            gamma=0.5,
            eta_max=0.1
        )
        return optimizer, lr_scheduler


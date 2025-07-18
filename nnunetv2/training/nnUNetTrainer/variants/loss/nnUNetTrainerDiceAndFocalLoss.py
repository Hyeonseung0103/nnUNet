import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_500epochs, nnUNetTrainer_2000epochs, nnUNetTrainerAdamCosine_2000epochs, nnUNetTrainerNAdam_2000epochs, nnUNetTrainerRAdam_2000epochs
from nnunetv2.training.loss.compound_losses import DC_and_FC_loss, DC_and_CE_loss
from nnunetv2.training.loss.focal import FocalLoss
from nnunetv2.hs_custom.lr_scheduler import CosineAnnealingWarmUpRestarts
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from torch.optim import AdamW, NAdam, RMSprop
import numpy as np

class nnUNetTrainerDiceAndFocalLoss(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossCosine(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossAdam(nnUNetTrainerAdamCosine_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossNAdam(nnUNetTrainerNAdam_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossNAdamCosine(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossRAdam(nnUNetTrainerRAdam_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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

class nnUNetTrainerDiceAndFocalLossRMSPropCosine(nnUNetTrainer_2000epochs):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # do_bg False 를 통해 background 는 loss 계산에서 제외 
        loss = DC_and_FC_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={},
            weight_focal=2, weight_dice=1, ignore_label=0
        )

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
        optimizer = RMSprop(self.network.parameters(),
                          momentum=0.9,
                          lr = 1e-2,
                          weight_decay=self.weight_decay)

        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=400,          # 첫 주기 길이(에폭 단위)
            T_mult=2,        # 주기 확장 계수 (T_mult=1: 고정 주기)
            eta_min=1e-6     # 최소 학습률
        )
        return optimizer, lr_scheduler

class nnUNetTrainerFocalLoss(nnUNetTrainer_500epochs):
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

# class nnUNetTrainerFocalLoss(nnUNetTrainer_2000epochs):
#     def _build_loss(self):
#         assert not self.label_manager.has_regions, "regions not supported by this trainer"
#         # do_bg False 를 통해 background 는 loss 계산에서 제외 
#         loss = DC_and_FC_loss(
#             soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
#                             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
#             focal_kwargs={},
#             weight_focal=2, weight_dice=1, ignore_label=0
#         )

#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         if self.enable_deep_supervision:
#             deep_supervision_scales = self._get_deep_supervision_scales()
#             weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
#             weights[-1] = 0

#             # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#             weights = weights / weights.sum()
#             # now wrap the loss
#             loss = DeepSupervisionWrapper(loss, weights)
#         return loss


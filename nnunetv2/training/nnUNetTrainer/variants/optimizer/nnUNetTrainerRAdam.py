from torch.optim import RAdam

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerRAdam(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = RAdam(self.network.parameters(),
                        #   lr=self.initial_lr,
                          lr = 1e-2,
                          weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, initial_lr=1e-2, max_steps=self.num_epochs)
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler
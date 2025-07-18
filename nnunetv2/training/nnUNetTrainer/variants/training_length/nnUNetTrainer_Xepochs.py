import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdam import nnUNetTrainerAdam
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerNAdam import nnUNetTrainerNAdam
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerRAdam import nnUNetTrainerRAdam
from nnunetv2.hs_custom.lr_scheduler import CosineAnnealingWarmUpRestarts
from torch.optim import Adam, AdamW

class nnUNetTrainer_5epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainer_1epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1


class nnUNetTrainer_10epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10


class nnUNetTrainer_20epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20


class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainer_100epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

class nnUNetTrainer_200epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200

class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainer_500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_750epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainer_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000


class nnUNetTrainerCosine_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)        
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=200,          # 첫 주기 길이(에폭 단위)
            T_mult=2,        # 주기 확장 계수 (T_mult=1: 고정 주기)
            T_up=10,
            gamma=0.5,
            eta_max=0.1
        )
        return optimizer, lr_scheduler


class nnUNetTrainerAdamCosine_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                                lr=self.initial_lr,
                                weight_decay=self.weight_decay,
                                amsgrad=True)        
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=200,          # 첫 주기 길이(에폭 단위)
            T_mult=2,        # 주기 확장 계수 (T_mult=1: 고정 주기)
            T_up=10,
            gamma=0.5,
            eta_max=0.1
        )
        return optimizer, lr_scheduler

class nnUNetTrainerNAdam_2000epochs(nnUNetTrainerNAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000

class nnUNetTrainerRAdam_2000epochs(nnUNetTrainerRAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
    
class nnUNetTrainer_4000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4000


class nnUNetTrainer_8000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 8000

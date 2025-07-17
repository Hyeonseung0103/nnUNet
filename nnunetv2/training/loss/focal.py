import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     '''
#     Multi-class Focal Loss
#     '''
#     def __init__(self, gamma=2, weight=None, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def forward(self, input, target):
#         """
#         input: [N, C], float32
#         target: [N, ], int64
#         """
#         logpt = F.log_softmax(input, dim=1)
#         pt = torch.exp(logpt)
#         logpt = (1-pt)**self.gamma * logpt
#         loss = F.nll_loss(logpt, target, self.weight)
#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # input: (B, C, H, W), target: (B, 1, H, W)
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0] # (B,H,W) 로 변경
        logpt = -F.cross_entropy(input, target.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(input.device)
            at = alpha_tensor.gather(0, target.long())
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import warnings

from torch.nn.modules.module import Module
from torch.nn import _reduction as _Reduction

import torch
from torch import Tensor
from typing import Callable, Optional

import segmentation_models_pytorch as smp
import torch.nn as nn


from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)
    

def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    mask = target == 0
    input[mask] = 0
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            mse_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction)) / torch.sum(~mask)


class Loss_function(nn.Module):
  def __init__(self, mode, losses, device):
    super().__init__()
    self.Dice = smp.losses.DiceLoss(mode=mode)
    self.Jaccard = smp.losses.JaccardLoss(mode=mode)
    self.Focal = smp.losses.FocalLoss(mode=mode)
    self.Lovasz = smp.losses.LovaszLoss(mode=mode)
    self.Tversky = smp.losses.TverskyLoss(mode=mode)
    self.losses_in_sum = []
    self.device = device
    for loss in losses:
        if loss == "dice":
            self.losses_in_sum.append(self.Dice)
        elif loss == "jaccard":
            self.losses_in_sum.append(self.Jaccard)
        elif loss == "lovasz":
            self.losses_in_sum.append(self.Lovasz)
        elif loss == "focal":
            self.losses_in_sum.append(self.Focal)
        elif loss == "tverscy":
            self.losses_in_sum.append(self.Tversky)


  def forward(self, predictions,  targets):
    loss = 0
    for loss_fn in self.losses_in_sum:
        loss += loss_fn(predictions, targets)
    return loss
from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric

from typing import Tuple
from torchmetrics.utilities.checks import _check_same_shape


class MeanSquaredError(Metric):
    r"""Computes `mean squared error`_ (MSE):

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MeanSquaredError
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> mean_squared_error = MeanSquaredError()
        >>> mean_squared_error(preds, target)
        tensor(0.8750)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)


def _mean_squared_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Squared Error.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    diff = preds - target
    mask = target == 0
    diff[mask] = 0
    sum_squared_error = torch.sum(diff * diff)
    # n_obs = target.numel()
    n_obs = torch.sum(~mask)
    return sum_squared_error, n_obs


def _mean_squared_error_compute(sum_squared_error: Tensor, n_obs: int, squared: bool = True) -> Tensor:
    """Computes Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_squared_error, n_obs = _mean_squared_error_update(preds, target)
        >>> _mean_squared_error_compute(sum_squared_error, n_obs)
        tensor(0.2500)
    """
    return sum_squared_error / n_obs if squared else torch.sqrt(sum_squared_error / n_obs)


def mean_squared_error(preds: Tensor, target: Tensor, squared: bool = True) -> Tensor:
    """Computes mean squared error.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False

    Return:
        Tensor with MSE

    Example:
        >>> from torchmetrics.functional import mean_squared_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_squared_error(x, y)
        tensor(0.2500)
    """
    sum_squared_error, n_obs = _mean_squared_error_update(preds, target)
    return _mean_squared_error_compute(sum_squared_error, n_obs, squared=squared)


class MeanAbsoluteError(Metric):
    r"""`Computes Mean Absolute Error`_ (MAE):

    .. math:: \text{MAE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} |

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import MeanAbsoluteError
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error = MeanAbsoluteError()
        >>> mean_absolute_error(preds, target)
        tensor(0.5000)
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)
    


def _mean_absolute_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Absolute Error.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()
    target = target if target.is_floating_point else target.float()

    mask = target == 0
    preds[mask] = 0

    sum_abs_error = torch.sum(torch.abs(preds - target))
    # n_obs = target.numel()
    n_obs = torch.sum(~mask)
    return sum_abs_error, n_obs


def _mean_absolute_error_compute(sum_abs_error: Tensor, n_obs: int) -> Tensor:
    """Computes Mean Absolute Error.

    Args:
        sum_abs_error: Sum of absolute value of errors over all observations
        n_obs: Number of predictions or observations

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)
        >>> _mean_absolute_error_compute(sum_abs_error, n_obs)
        tensor(0.2500)
    """

    return sum_abs_error / n_obs


def mean_absolute_error(preds: Tensor, target: Tensor) -> Tensor:
    """Computes mean absolute error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAE

    Example:
        >>> from torchmetrics.functional import mean_absolute_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_absolute_error(x, y)
        tensor(0.2500)
    """
    sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)
    return _mean_absolute_error_compute(sum_abs_error, n_obs)

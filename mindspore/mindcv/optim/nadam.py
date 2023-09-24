"""nadam"""
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    assert isinstance(beta1, float) and 0 <= beta1 <= 1.0, f"For {prim_name}, beta1 should between 0 and 1"
    assert isinstance(beta2, float) and 0 <= beta2 <= 1.0, f"For {prim_name}, beta2 should between 0 and 1"
    assert isinstance(eps, float) and eps > 0, f"For {prim_name}, eps should be bigger than 0"


_scaler_one = Tensor(1, ms.float32)


class NAdam(Optimizer):
    """
    Implements NAdam algorithm (a variant of Adam based on Nesterov momentum).
    """

    @opt_init_args_register
    def __init__(
        self,
        params,
        learning_rate=2e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        loss_scale=1.0,
        schedule_decay=4e-3,
    ):
        super().__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="nadam_m", init="zeros")
        self.moments2 = self.parameters.clone(prefix="nadam_v", init="zeros")
        self.schedule_decay = Tensor(np.array([schedule_decay]).astype(np.float32))
        self.mu_schedule = Parameter(initializer(1, [1], ms.float32), name="mu_schedule")
        self.beta2_power = Parameter(initializer(1, [1], ms.float32), name="beta2_power")

    def construct(self, gradients):
        lr = self.get_lr()
        params = self.parameters
        step = self.global_step + _scaler_one
        gradients = self.decay_weight(gradients)
        mu = self.beta1 * (
            _scaler_one - Tensor(0.5, ms.float32) * ops.pow(Tensor(0.96, ms.float32), step * self.schedule_decay)
        )
        mu_next = self.beta1 * (
            _scaler_one
            - Tensor(0.5, ms.float32) * ops.pow(Tensor(0.96, ms.float32), (step + _scaler_one) * self.schedule_decay)
        )
        mu_schedule = self.mu_schedule * mu
        mu_schedule_next = self.mu_schedule * mu * mu_next
        self.mu_schedule = mu_schedule
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        num_params = len(params)
        for i in range(num_params):
            ops.assign(self.moments1[i], self.beta1 * self.moments1[i] + (_scaler_one - self.beta1) * gradients[i])
            ops.assign(
                self.moments2[i], self.beta2 * self.moments2[i] + (_scaler_one - self.beta2) * ops.square(gradients[i])
            )

            regulate_m = mu_next * self.moments1[i] / (_scaler_one - mu_schedule_next) + (_scaler_one - mu) * gradients[
                i
            ] / (_scaler_one - mu_schedule)
            regulate_v = self.moments2[i] / (_scaler_one - beta2_power)

            update = params[i] - lr * regulate_m / (self.eps + ops.sqrt(regulate_v))
            ops.assign(params[i], update)

        return params

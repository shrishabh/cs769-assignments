import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m_zero'] = torch.zeros(group['params'][idx].size())
                    state['v_zero'] = torch.zeros(group['params'][idx].size())
                    # state['m_zero'] = torch.zeros(grad.size())
                    # state['v_zero'] = torch.zeros(grad.size())
                t = state['t'] + 1
                state['t'] = t
                m_zero = state['m_zero']
                v_zero = state['v_zero']


                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group['betas']
                epsilon = group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']

                # Update first and second moments of the gradients
                m_zero = beta1*state['m_zero'] + (1-beta1)*grad
                v_zero = beta2*state['v_zero'] + (1-beta2)*grad*grad

                state['m_zero'] = m_zero
                state['v_zero'] = v_zero

                # state['m_zero'].mul_(beta1).addcmul_(grad, value=1.0-beta1)
                # state['v_zero'].mul_(beta2).addcmul_(grad, grad, value=1.0-beta2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if correct_bias:
                    m_hat = state['m_zero']/(1-(beta1)**t)
                    v_hat = state['v_zero']/(1-(beta2)**t)
                else:
                    m_hat = state['m_zero']
                    v_hat = state['v_zero']

                # Update parameters
                update = alpha*m_hat/(torch.sqrt(v_hat) + epsilon)
                group['params'][idx].data = group['params'][idx] - update

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                group['params'][idx].data = group['params'][idx].data - alpha*weight_decay*group['params'][idx].data

        return loss

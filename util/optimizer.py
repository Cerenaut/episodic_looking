import logging
from dataclasses import dataclass

import torch
from torch import optim

logger = logging.getLogger(__name__)

@dataclass
class ModelOptimizerConfig:

    name: str = "Optimizer"
    optimizer_type: str = "adam"
    learning_rate: float = 0.01
    momentum: float = 0.0  # SGD only
    weight_decay: float = 0.01  # Adam/AdamW only. Reported reasonable default
    clip_grad_norm: float = 0.0  # try 0.5, 1.0


class ModelOptimizer:

    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_ADAMW = "adamw"
    OPTIMIZER_SGD = "sgd"
    
    def __init__(self, config, parameters):
        self.config = config

        # NB: Parameters only iterable once
        logging.info(f"Optimizer {self.config.name}: {self.config.optimizer_type}")
        if self.config.optimizer_type == ModelOptimizer.OPTIMIZER_ADAM:
            if self.config.weight_decay > 0:
                self.optimizer = optim.Adam(
                    parameters, 
                    lr=self.config.learning_rate,
                    weight_decay = self.config.weight_decay  # really, L2 regularization
                )
            else:                   
                self.optimizer = optim.Adam(
                    parameters, 
                    lr=self.config.learning_rate,
                )
        elif self.config.optimizer_type == ModelOptimizer.OPTIMIZER_ADAMW:
            self.optimizer = optim.Adam(
                parameters, 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,  # weight decay ignores adaptive learning rate
            )
        elif self.config.optimizer_type == ModelOptimizer.OPTIMIZER_SGD:
            self.optimizer = optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.config.name} type not recognized: {self.config.optimizer}")            

    def clear_gradients(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimize(self, loss, parameters):
        self.clear_gradients()
        loss.backward()
        self.clip_gradients(parameters)
        self.optimizer.step()

    def clip_gradients(self, parameters):
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, 
                self.config.clip_grad_norm,
            )

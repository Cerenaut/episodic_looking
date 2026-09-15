import logging

import torch
from torch import nn

from model.sparse.sparse_dense_model import (
    SparseActivationDenseModel,
    SparseActivationDenseModelConfig,
)
from model.sparse.sparse_distributed_encoder import SparseDistributedEncoder

logger = logging.getLogger(__name__)

class SparseDistributedModel(SparseDistributedEncoder):
    """
    Sparsely activated cliques of hidden cells avoid interference, allowing fast learning.
    Values learn using only these cliques and are represented via dense fully-connected 
    feed-forward models.

    Multiple models are bundled together so they can share the same Encoder as the top-k 
    is relatively expensive.
    """

    def __init__(
        self,
        input_key_size: int,
        input_value_size:int,
        memory_size: int,
        ensemble_size: int,
        sparsity: int,
        model_configs: list[SparseActivationDenseModelConfig],
    ):
        logger.info(f"Key size:{input_key_size} Input size:{input_value_size} Memory size:{memory_size} Ensemble size:{ensemble_size} Sparsity:{sparsity}")
        super().__init__(
            input_key_size = input_key_size,
            memory_size = memory_size,
            ensemble_size = ensemble_size,
            sparsity = sparsity,
        )
        self.input_value_size = input_value_size        
        self.model_configs = model_configs
        self.trainable_modules = nn.ModuleDict()
        self.create_models()

    def create_models(self):
        for model_index, model_config in enumerate(self.model_configs):
            model_name = self.get_model_name(model_index=model_index)
            logger.info(f"Model #:{model_index} Name:{model_name}")

            assert(model_config.hidden_size == self.memory_size * model_config.hidden_size_factor)
            model = SparseActivationDenseModel(model_config)
            self.trainable_modules[model_name] = model

    def get_model_name(self, model_index:int) -> str:
        model_config = self.model_configs[model_index]
        return model_config.name
    
    def get_model(self, model_index:int):
        model_name = self.get_model_name(model_index=model_index)
        model = self.trainable_modules[model_name] 
        return model

    def do_model(
        self, 
        key_input:torch.Tensor, 
        model_input:torch.Tensor, 
        model_index:int = 0,
        active_mask:torch.Tensor|None = None,
    ):
        active_indices = self.encode(key_input)  # returns [B, sparsity]

        model = self.get_model(model_index)
        if active_mask is None:
            active_mask = model.create_active_mask(
                model_input,  # only used for value dtype and mask
                active_indices=active_indices,
            )

        output = model(model_input, active_mask = active_mask)
        return output, active_mask  # can reuse mask with other models

    def get_trainable_parameters(self) -> list:
        models_parameters = []
        for model_index, model_config in enumerate(self.model_configs):
            parameters = self.get_model_parameters(model_index)
            models_parameters.extend(parameters)
        return models_parameters

    def get_model_parameters(self, model_index:int = 0):
        model_name = self.get_model_name(model_index=model_index)
        parameters_list = list(self.trainable_modules[model_name].parameters())
        return parameters_list

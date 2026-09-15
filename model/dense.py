from dataclasses import dataclass

from torch import nn

from util.device import get_device
from util.model import create_functional_nonlinearity


@dataclass
class DenseModelConfig:

    name: str = "Dense"
    nonlinearity: str = "leaky-relu"
    input_layer_norm: bool = False
    input_dropout: float = 0.0
    input_weight_clip: float = 0.0  # e.g. 0.1
    input_size: int = 0
    hidden_size: int = 0
    output_size: int = 0
    output_nonlinearity: str = "leaky-relu"
    layers: int = 0
    hidden_dropout: float = 0.0
    bias: bool = True

class DenseModel(nn.Module):
    """
    Dense (fully connected) network multi-layer network with regularization features such as dropout. 
    Outputs logits from hidden layer. This is just a wrapper around Torch APIs which constructs the 
    model from a dataclass config with all the feature options needed for experimentation.
    """

    def __init__(self, config):
        super().__init__()  # Must be first to auto detect trainable parameters

        self.device = get_device()
        self.config = config
        self.trainable_modules = nn.ModuleDict()

        if self.config.input_layer_norm:
            self.ln = nn.LayerNorm(self.config.input_size, device=self.device)
            ln_name = self.get_module_name("dense-input-layer-norm")
            self.trainable_modules[ln_name] = self.ln
        else:
            self.ln = None

        if self.config.input_dropout > 0:
            self.dropout_input = nn.Dropout(p=config.input_dropout)
            dropout_name = self.get_module_name("dense-input-dropout")
            self.trainable_modules[dropout_name] = self.dropout_input
        else:
            self.dropout_input = None

        self.layers = []
        self.dropout_layers = []
        for layer in range(self.config.layers):
            if layer == 0: 
                x = self.config.input_size
            else:
                x = self.config.hidden_size

            if layer < (self.config.layers -1):
                y = self.config.hidden_size
            else:  # last layer
                y = self.config.output_size

            fc = nn.Linear(x, y, bias=self.config.bias, device=self.device)
            layer_name = self.get_module_name("dense-layer-"+str(layer))
            self.trainable_modules[layer_name] = fc
            self.layers.append(fc)

            # First layer
            layer_index = len(self.layers) -1
            self._apply_weight_clip(layer_index)

            # Last layer
            if layer < (self.config.layers -1):  # no dropout on output layer
                if self.config.hidden_dropout > 0:
                    d = nn.Dropout(p=config.hidden_dropout)
                    dropout_name = self.get_module_name("d-"+str(layer))
                    self.dropout_layers.append(d)
                    self.trainable_modules[dropout_name] = d

        # Create nonlinearities
        self.f_hidden = create_functional_nonlinearity(self.config.nonlinearity)
        self.f_output = create_functional_nonlinearity(self.config.output_nonlinearity)        

    def _apply_weight_clip(self, layer:int):
        # Apply an L1 penalty or explicit weight clipping on the input weights:
        if layer == 0:
            if self.config.input_weight_clip > 0:
                fc = self.layers[layer]
                for param in fc.parameters():
                    param.data.clamp_(-self.config.input_weight_clip, self.config.input_weight_clip)

    def reset_parameters(self):
        if self.ln is not None:
            self.ln.reset_parameters()
            
        for layer_index, layer in enumerate(self.layers):
            layer.reset_parameters() 
            self._apply_weight_clip(layer_index)

    def forward(self, x):
        if self.ln is not None:
            x = self.ln(x)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        for layer in range(self.config.layers):
            fc = self.layers[layer]
            x = fc(x)

            f = self.f_output
            if layer < (self.config.layers -1):  # if not output layer
                f = self.f_hidden

            if f is not None:
                x = f(x)

            if layer < (self.config.layers -1):  # if not output layer
                if self.config.hidden_dropout > 0:  
                    d = self.dropout_layers[layer]
                    x = d(x)
        return x

    def get_module_name(self, suffix):
        if self.config.name != "":
            return self.config.name + "_" + suffix
        return suffix  # no name

    def __str__(self):
        return f"Name: {self.config.name} layers: {self.config.layers} sizes in: {self.config.input_size} hidden: {self.config.hidden_size} out: {self.config.output_size}"

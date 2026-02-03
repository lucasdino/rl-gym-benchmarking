import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SingleNetworkConfig
from networks.helper import LINEAR_LAYER_MAP, ACTIV_MAP



class MLP(nn.Module):
    def __init__(self, cfg: SingleNetworkConfig, obs_space, act_space):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        mlp_dims = cfg.network_args["mlp_dims"]
        activations = cfg.network_args["mlp_activations"]
        bias = cfg.network_args["mlp_bias"]
        layer_types = cfg.network_args["mlp_layer_type"]
        layer_extra_args = cfg.network_args["mlp_layer_extra_args"]

        layer_shapes = []
        for dim in mlp_dims:
            if dim == "InDims":
                layer_shapes.append(math.prod(obs_space.shape))
            elif dim == "OutDims":
                layer_shapes.append(act_space.n)
            else:
                layer_shapes.append(int(dim))

        if len(layer_types) != len(layer_shapes) - 1:
            raise ValueError("mlp_layer_type length must match number of layers")
        if len(layer_extra_args) != len(layer_shapes) - 1:
            raise ValueError("mlp_layer_extra_args length must match number of layers")

        # Setting up our various layers
        layers = []
        for i in range(len(layer_shapes) - 1):
            layer_cls = LINEAR_LAYER_MAP[layer_types[i]]
            extra_args = layer_extra_args[i]
            if not isinstance(extra_args, dict):
                raise TypeError("mlp_layer_extra_args must be a list of dicts")
            layers.append(layer_cls(layer_shapes[i], layer_shapes[i+1], bias=bias, **extra_args))
            if i < len(activations):
                layers.append(ACTIV_MAP[activations[i]]())

        # Defining our final network
        self.net = nn.Sequential(*layers)
        self._print_architecture()


    def _init_params(self):
        """ Common RL init: orthogonal + ReLU gain, small uniform last layer. """
        linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        last = linear_layers[-1]
        for m in linear_layers:
            if m is last:
                nn.init.uniform_(m.weight, -1e-3, 1e-3)
            else:
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def _format_layer(self, layer: nn.Module) -> str:
        """Return a readable layer description."""
        name = layer.__class__.__name__
        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            return f"{name}({layer.in_features}->{layer.out_features})"
        return name

    def _print_architecture(self) -> None:
        """Print a concise architecture summary."""
        parts = [self._format_layer(layer) for layer in self.net]
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[Net: {self.cfg.name}] " + " -> ".join(parts) + f" | Params: {num_params:,}")

    def forward(self, x):
        return self.net(x)
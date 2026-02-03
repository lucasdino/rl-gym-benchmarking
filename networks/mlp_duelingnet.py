import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SingleNetworkConfig
from networks.helper import LINEAR_LAYER_MAP, ACTIV_MAP



class MLP_DuelingNet(nn.Module):
    def __init__(self, cfg: SingleNetworkConfig, obs_space, act_space):
        """ Implementing from this paper: https://arxiv.org/pdf/1511.06581. """
        super(MLP_DuelingNet, self).__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        shared_dims = cfg.network_args["shared_mlp_dims"]
        shared_activations = cfg.network_args["shared_mlp_activations"]
        shared_layer_types = cfg.network_args["shared_mlp_layer_type"]
        shared_layer_extra_args = cfg.network_args["shared_mlp_layer_extra_args"]

        value_dims = cfg.network_args["valuehead_mlp_dims"]
        value_activations = cfg.network_args["valuehead_mlp_activations"]
        value_layer_types = cfg.network_args["valuehead_mlp_layer_type"]
        value_layer_extra_args = cfg.network_args["valuehead_mlp_layer_extra_args"]

        action_dims = cfg.network_args["actionhead_mlp_dims"]
        action_activations = cfg.network_args["actionhead_mlp_activations"]
        action_layer_types = cfg.network_args["actionhead_mlp_layer_type"]
        action_layer_extra_args = cfg.network_args["actionhead_mlp_layer_extra_args"]

        bias = cfg.network_args["mlp_bias"]

        self.shared_net, shared_out_dim = self._build_shared(
            shared_dims,
            shared_activations,
            shared_layer_types,
            shared_layer_extra_args,
            bias,
        )
        self.value_head = self._build_head(
            value_dims,
            value_activations,
            value_layer_types,
            value_layer_extra_args,
            bias,
            in_dim=shared_out_dim,
            out_dim=1,
        )
        self.action_head = self._build_head(
            action_dims,
            action_activations,
            action_layer_types,
            action_layer_extra_args,
            bias,
            in_dim=shared_out_dim,
            out_dim=act_space.n,
        )

        self._print_architecture()


    def _init_params(self):
        """ Common RL init: orthogonal + ReLU gain, small uniform last layer. """
        linear_layers = [m for m in self.shared_net.modules() if isinstance(m, nn.Linear)]
        linear_layers += [m for m in self.value_head.modules() if isinstance(m, nn.Linear)]
        linear_layers += [m for m in self.action_head.modules() if isinstance(m, nn.Linear)]
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
        shared_parts = [self._format_layer(layer) for layer in self.shared_net]
        value_parts = [self._format_layer(layer) for layer in self.value_head]
        action_parts = [self._format_layer(layer) for layer in self.action_head]
        num_params = sum(p.numel() for p in self.parameters())
        msg = f"[Net: {self.cfg.name}] "
        msg += "Shared: " + " -> ".join(shared_parts)
        msg += " | Value: " + " -> ".join(value_parts)
        msg += " | Action: " + " -> ".join(action_parts)
        msg += f" | Params: {num_params:,}"
        print(msg)

    def forward(self, x):
        shared = self.shared_net(x)
        value = self.value_head(shared)
        action = self.action_head(shared)
        action_mean = action.mean(dim=1, keepdim=True)
        return value + (action - action_mean)

    def _build_shared(
        self,
        dims,
        activations,
        layer_types,
        layer_extra_args,
        bias,
    ):
        """Build shared trunk."""
        layer_shapes = []
        for dim in dims:
            if dim == "InDims":
                layer_shapes.append(math.prod(self.obs_space.shape))
            elif dim == "OutDims":
                raise ValueError("shared_mlp_dims cannot use OutDims")
            else:
                layer_shapes.append(int(dim))

        if len(layer_types) != len(layer_shapes) - 1:
            raise ValueError("shared_mlp_layer_type length must match number of layers")
        if len(layer_extra_args) != len(layer_shapes) - 1:
            raise ValueError("shared_mlp_layer_extra_args length must match number of layers")

        layers = []
        for i in range(len(layer_shapes) - 1):
            layer_cls = LINEAR_LAYER_MAP[layer_types[i]]
            extra_args = layer_extra_args[i]
            if not isinstance(extra_args, dict):
                raise TypeError("shared_mlp_layer_extra_args must be a list of dicts")
            layers.append(layer_cls(layer_shapes[i], layer_shapes[i + 1], bias=bias, **extra_args))
            if i < len(activations):
                layers.append(ACTIV_MAP[activations[i]]())

        return nn.Sequential(*layers), layer_shapes[-1]

    def _build_head(
        self,
        dims,
        activations,
        layer_types,
        layer_extra_args,
        bias,
        in_dim,
        out_dim,
    ):
        """Build a head network."""
        layer_shapes = []
        for dim in dims:
            if dim == "InDims":
                layer_shapes.append(in_dim)
            elif dim == "OutDims":
                layer_shapes.append(out_dim)
            else:
                layer_shapes.append(int(dim))

        if len(layer_types) != len(layer_shapes) - 1:
            raise ValueError("head_mlp_layer_type length must match number of layers")
        if len(layer_extra_args) != len(layer_shapes) - 1:
            raise ValueError("head_mlp_layer_extra_args length must match number of layers")

        layers = []
        for i in range(len(layer_shapes) - 1):
            layer_cls = LINEAR_LAYER_MAP[layer_types[i]]
            extra_args = layer_extra_args[i]
            if not isinstance(extra_args, dict):
                raise TypeError("head_mlp_layer_extra_args must be a list of dicts")
            layers.append(layer_cls(layer_shapes[i], layer_shapes[i + 1], bias=bias, **extra_args))
            if i < len(activations):
                layers.append(ACTIV_MAP[activations[i]]())

        return nn.Sequential(*layers)
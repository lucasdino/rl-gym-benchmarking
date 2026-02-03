import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SingleNetworkConfig
from networks.helper import LINEAR_LAYER_MAP, ACTIV_MAP



class CNN_DistributionalDueling(nn.Module):
    """
    Combo of dueling nets (https://arxiv.org/pdf/1511.06581) and
    distributional rl (https://arxiv.org/pdf/1707.06887) with CNN backbone.
    """
    def __init__(self, cfg: SingleNetworkConfig, obs_space, act_space):
        super(CNN_DistributionalDueling, self).__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        self.num_atoms = int(cfg.network_args["dist_num_atoms"])

        cnn_layers_cfg = cfg.network_args["cnn_layers"]
        cnn_activations = cfg.network_args.get("cnn_activations", ["relu"] * len(cnn_layers_cfg))
        input_format = cfg.network_args.get("input_format", "infer")

        value_dims = cfg.network_args["valuehead_mlp_dims"]
        value_activations = cfg.network_args["valuehead_mlp_activations"]
        value_layer_types = cfg.network_args["valuehead_mlp_layer_type"]
        value_layer_extra_args = cfg.network_args["valuehead_mlp_layer_extra_args"]

        action_dims = cfg.network_args["actionhead_mlp_dims"]
        action_activations = cfg.network_args["actionhead_mlp_activations"]
        action_layer_types = cfg.network_args["actionhead_mlp_layer_type"]
        action_layer_extra_args = cfg.network_args["actionhead_mlp_layer_extra_args"]

        bias = cfg.network_args["mlp_bias"]

        in_channels, height, width, self._input_format = self._resolve_input_shape(obs_space.shape, input_format)
        in_channels_frozen = in_channels

        conv_layers = []
        for idx, layer_cfg in enumerate(cnn_layers_cfg):
            out_channels = int(layer_cfg["out_channels"])
            kernel_size = layer_cfg["kernel_size"]
            stride = layer_cfg["stride"]
            padding = layer_cfg.get("padding", 0)
            dilation = layer_cfg.get("dilation", 1)

            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
            if idx < len(cnn_activations):
                conv_layers.append(ACTIV_MAP[cnn_activations[idx]]())
            in_channels = out_channels

        self.cnn = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels_frozen, height, width)
            conv_out = self.cnn(dummy)
            conv_out_dim = int(conv_out.view(1, -1).shape[1])

        self.value_head = self._build_head(
            value_dims,
            value_activations,
            value_layer_types,
            value_layer_extra_args,
            bias,
            in_dim=conv_out_dim,
            out_dim=self.num_atoms,
        )
        self.action_head = self._build_head(
            action_dims,
            action_activations,
            action_layer_types,
            action_layer_extra_args,
            bias,
            in_dim=conv_out_dim,
            out_dim=act_space.n * self.num_atoms,
        )

        self._print_architecture()

    def _init_params(self):
        """ Common RL init: orthogonal + ReLU gain, small uniform last layer. """
        linear_layers = [m for m in self.cnn.modules() if isinstance(m, nn.Linear)]
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

    def forward(self, x):
        if self._input_format == "HWC":
            x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = x.contiguous().view(x.shape[0], -1)

        value = self.value_head(x).view(-1, 1, self.num_atoms)
        action = self.action_head(x).view(-1, self.act_space.n, self.num_atoms)
        action_mean = action.mean(dim=1, keepdim=True)
        return value + (action - action_mean)

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
        """Build a head network (value or action)."""
        layer_shapes = []
        for dim in dims:
            if dim == "InDims":
                layer_shapes.append(in_dim)
            elif dim == "OutDims":
                layer_shapes.append(out_dim)
            else:
                layer_shapes.append(int(dim))

        layer_shapes[-1] = out_dim

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

    def _format_layer(self, layer: nn.Module) -> str:
        """Return a readable layer description."""
        name = layer.__class__.__name__
        if isinstance(layer, nn.Conv2d):
            return f"{name}({layer.in_channels}->{layer.out_channels}, k={layer.kernel_size}, s={layer.stride})"
        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            return f"{name}({layer.in_features}->{layer.out_features})"
        return name

    def _print_architecture(self) -> None:
        """Print a concise architecture summary."""
        cnn_parts = [self._format_layer(layer) for layer in self.cnn]
        value_parts = [self._format_layer(layer) for layer in self.value_head]
        action_parts = [self._format_layer(layer) for layer in self.action_head]
        num_params = sum(p.numel() for p in self.parameters())
        msg = f"[Net: {self.cfg.name}] "
        msg += "CNN: " + " -> ".join(cnn_parts)
        msg += " | Value: " + " -> ".join(value_parts)
        msg += " | Action: " + " -> ".join(action_parts)
        msg += f" | Params: {num_params:,}"
        print(msg)

    @staticmethod
    def _resolve_input_shape(shape: tuple[int, ...], input_format: str) -> tuple[int, int, int, str]:
        if len(shape) != 3:
            raise ValueError("CNN expects a 3D obs_space shape")
        if input_format == "CHW":
            return int(shape[0]), int(shape[1]), int(shape[2]), "CHW"
        if input_format == "HWC":
            return int(shape[2]), int(shape[0]), int(shape[1]), "HWC"
        raise ValueError("input_format must be 'CHW', 'HWC', or 'infer'")
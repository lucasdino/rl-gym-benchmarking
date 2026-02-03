import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SingleNetworkConfig
from networks.helper import LINEAR_LAYER_MAP, ACTIV_MAP



class CNN(nn.Module):
    def __init__(self, cfg: SingleNetworkConfig, obs_space, act_space):
        super(CNN, self).__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        cnn_layers_cfg = cfg.network_args["cnn_layers"]
        cnn_activations = cfg.network_args.get("cnn_activations", ["relu"] * len(cnn_layers_cfg))
        input_format = cfg.network_args.get("input_format", "infer")

        mlp_dims = cfg.network_args["mlp_dims"]
        mlp_activations = cfg.network_args["mlp_activations"]
        mlp_bias = cfg.network_args["mlp_bias"]
        mlp_layer_types = cfg.network_args["mlp_layer_type"]
        mlp_layer_extra_args = cfg.network_args["mlp_layer_extra_args"]

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

        mlp_shapes = []
        for dim in mlp_dims:
            if dim == "InDims":
                mlp_shapes.append(conv_out_dim)
            elif dim == "OutDims":
                mlp_shapes.append(act_space.n)
            else:
                mlp_shapes.append(int(dim))

        if len(mlp_layer_types) != len(mlp_shapes) - 1:
            raise ValueError("mlp_layer_type length must match number of layers")
        if len(mlp_layer_extra_args) != len(mlp_shapes) - 1:
            raise ValueError("mlp_layer_extra_args length must match number of layers")

        mlp_layers = []
        for i in range(len(mlp_shapes) - 1):
            layer_cls = LINEAR_LAYER_MAP[mlp_layer_types[i]]
            extra_args = mlp_layer_extra_args[i]
            if not isinstance(extra_args, dict):
                raise TypeError("mlp_layer_extra_args must be a list of dicts")
            mlp_layers.append(layer_cls(mlp_shapes[i], mlp_shapes[i+1], bias=mlp_bias, **extra_args))
            if i < len(mlp_activations):
                mlp_layers.append(ACTIV_MAP[mlp_activations[i]]())

        self.head = nn.Sequential(*mlp_layers)
        # print(f"Obs Space: {self.obs_space}")
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


    def forward(self, x):
        if self._input_format == "HWC":
            x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = x.contiguous().view(x.shape[0], -1)
        return self.head(x)


    # ==================================================
    # Various QoL
    # ==================================================
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
        parts = [self._format_layer(layer) for layer in list(self.cnn) + list(self.head)]
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[Net: {self.cfg.name}] " + " -> ".join(parts) + f" | Params: {num_params:,}")

    @staticmethod
    def _resolve_input_shape(shape: tuple[int, ...], input_format: str) -> tuple[int, int, int, str]:
        if len(shape) != 3:
            raise ValueError("CNN expects a 3D obs_space shape")
        if input_format == "CHW":
            return int(shape[0]), int(shape[1]), int(shape[2]), "CHW"
        if input_format == "HWC":
            return int(shape[2]), int(shape[0]), int(shape[1]), "HWC"
        raise ValueError("input_format must be 'CHW', 'HWC', or 'infer'")
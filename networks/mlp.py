import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SingleNetworkConfig
from networks.helper.activations import ACTIV_MAP



class MLP(nn.Module):
    def __init__(self, cfg: SingleNetworkConfig, obs_space, act_space):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        hidden_dims = cfg.network_args["hidden_dims"]
        activations = cfg.network_args.get("activations", [])
        bias = cfg.network_args.get("bias", True)

        layer_shapes = [math.prod(obs_space.shape)] + hidden_dims + [act_space.n]

        # Setting up our various layers
        layers = []
        for i in range(len(layer_shapes) - 2):
            layers.append(nn.Linear(layer_shapes[i], layer_shapes[i+1], bias=bias))
            if i < len(activations):
                layers.append(ACTIV_MAP[activations[i]]())
        layers.append(nn.Linear(layer_shapes[-2], layer_shapes[-1]))

        # Defining our final network
        self.net = nn.Sequential(*layers)

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
        return self.net(x)
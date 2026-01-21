from networks.mlp import MLP

from configs.config import SingleNetworkConfig



NETWORK_MAPPING = {
    "mlp": MLP
}

def build_network(cfg: SingleNetworkConfig, obs_space, act_space):
    network = NETWORK_MAPPING[cfg.network_type]
    return network(
        cfg = cfg, 
        obs_space = obs_space,
        act_space = act_space
    )
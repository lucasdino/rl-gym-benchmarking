from networks.mlp import MLP
from networks.cnn import CNN
from networks.mlp_duelingnet import MLP_DuelingNet
from networks.mlp_distributionaldueling import MLP_DistributionalDueling

from configs.config import SingleNetworkConfig



NETWORK_MAPPING = {
    "mlp": MLP,
    "cnn": CNN,
    "duelingnet": MLP_DuelingNet,
    "distributional_dueling": MLP_DistributionalDueling,

}

def build_network(cfg: SingleNetworkConfig, obs_space, act_space):
    network = NETWORK_MAPPING[cfg.network_type]
    return network(
        cfg = cfg, 
        obs_space = obs_space,
        act_space = act_space
    )
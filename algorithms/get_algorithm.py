from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.distributional_ddqn import Distributional_QN

from algorithms.sac import SoftActorCritic


ALGO_MAP = {
    "ddqn": DDQN,
    "dqn": DQN,
    "distributional_ddqn": Distributional_QN,
}

def get_algorithm(algorithm_name):
    return ALGO_MAP[algorithm_name]
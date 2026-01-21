from algorithms.sac import SoftActorCritic


ALGO_MAP = {
    "sac": SoftActorCritic
}


def get_algorithm(algorithm_name):
    return ALGO_MAP[algorithm_name]
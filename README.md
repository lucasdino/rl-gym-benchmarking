# Reinforcement Learning Benchmarking on Gymnasium Tasks   
Goal is for me to benchmark a ton of different RL algorithms including:
- Policy gradient, model-free baselines (PPO, SAC)
- Value-based, model-free baseline (Rainbow -- this is DDQN + all the tricks)
- Model-based SoTA methods that are newer and data efficient (TD-MPC2, DreamerV3)

Some of this won't be entirely true to the paper and we'll see how it changes (e.g., I'm not training a video gen action model for DreamerV3 but we could train a 'transition function' model on a simple task.)
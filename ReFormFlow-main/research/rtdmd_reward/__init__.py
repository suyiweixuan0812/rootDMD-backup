"""Independent reward-pair experiments for the attached RTDMD paper."""

from .pair_objective import pair_reward, pair_reward_gradients

__all__ = ["pair_reward", "pair_reward_gradients"]

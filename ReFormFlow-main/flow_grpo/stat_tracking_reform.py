import numpy as np
import torch
from scipy.special import logsumexp


class PerPromptStatTracker:

    def __init__(self, global_std=False, reward_clip=5.0):
        self.global_std = global_std
        self.reward_clip = reward_clip
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo'):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)

        if rewards.ndim > 1:
            rewards = rewards.mean(axis=-1)

        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))

        for prompt in unique:
            self.stats[prompt] = np.array(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt])
            if self.global_std:
                std = np.std(rewards) + 1e-4
            else:
                std = np.std(self.stats[prompt]) + 1e-4

            if type == 'grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == 'rwr':
                advantages[prompts == prompt] = prompt_rewards
            elif type == 'sft':
                t = torch.tensor(prompt_rewards)
                advantages[prompts == prompt] = (t == t.max()).float().numpy()

        return advantages

    def compute_reform_targets(self, prompts, rewards, beta=1.0, center=True):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)

        if rewards.ndim > 1:
            rewards = rewards.mean(axis=-1)

        N = len(rewards)
        advantages = np.zeros(N, dtype=np.float64)
        log_z = np.zeros(N, dtype=np.float64)
        scaled_rewards = np.zeros(N, dtype=np.float64)

        unique_prompts = np.unique(prompts)

        for prompt in unique_prompts:
            mask = prompts == prompt
            r_group = rewards[mask]
            G = len(r_group)

            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(r_group.tolist())
            self.history_prompts.add(hash(prompt))

            hist = np.array(self.stats[prompt])
            mean = hist.mean()
            if self.global_std:
                std = rewards.std() + 1e-8
            else:
                std = hist.std() + 1e-8

            if center:
                adv_group = (r_group - mean) / std
            else:
                adv_group = r_group / std
            adv_clipped = np.clip(adv_group, -self.reward_clip, self.reward_clip)
            beta_r = beta * adv_clipped
            log_z_val = logsumexp(beta_r) - np.log(G)

            advantages[mask] = adv_group
            scaled_rewards[mask] = beta_r
            log_z[mask] = log_z_val

        return advantages, log_z, scaled_rewards

    def get_stats(self):
        avg_group_size = (
            sum(len(v) if isinstance(v, list) else len(v.tolist())
                for v in self.stats.values()) / len(self.stats)
            if self.stats else 0
        )
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.rewards.base_reward import BaseReward


@registry.register_reward("penalized_reward")
class PenalizedReward(BaseReward):
    def __init__(self, reward, penalty, **kwargs):
        super().__init__(**kwargs)
        self.reward = reward
        self.penalty = penalty
        self.base_reward_name = reward.__class__.__name__
        self.penalty_name = penalty.__class__.__name__

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or {}
        if hasattr(cfg, "items"):
            cfg = dict(cfg.items())

        reward_cfg = cfg.get("reward")
        penalty_cfg = cfg.get("penalty")
        if reward_cfg is None or penalty_cfg is None:
            raise ValueError("PenalizedReward requires nested reward and penalty configs.")

        if hasattr(reward_cfg, "items"):
            reward_cfg = dict(reward_cfg.items())
        if hasattr(penalty_cfg, "items"):
            penalty_cfg = dict(penalty_cfg.items())

        reward_name = reward_cfg["name"]
        reward_cls = registry.get_reward_class(reward_name)
        if reward_cls is None:
            raise ValueError(f"Reward {reward_name} is not registered.")
        reward_kwargs = {k: v for k, v in reward_cfg.items() if k != "name"}
        reward = reward_cls.from_config(reward_kwargs)

        penalty_name = penalty_cfg["name"]
        penalty_cls = registry.get_penalty_class(penalty_name)
        if penalty_cls is None:
            raise ValueError(f"Penalty {penalty_name} is not registered.")
        penalty_kwargs = {k: v for k, v in penalty_cfg.items() if k != "name"}
        penalty = penalty_cls.from_config(penalty_kwargs)

        kwargs = {k: v for k, v in cfg.items() if k not in {"reward", "penalty"}}
        return cls(reward=reward, penalty=penalty, **kwargs)

    def score(self, samples, responses, reward_metas):
        base_reward_tensor, reward_info = self.reward.score(samples, responses, reward_metas)
        if not torch.is_tensor(base_reward_tensor):
            base_reward_tensor = torch.tensor(base_reward_tensor, dtype=torch.float32)

        penalty_tensor, penalty_info = self.penalty.compute(
            samples=samples,
            responses=responses,
            reward_metas=reward_metas,
            reward_tensor=base_reward_tensor,
            reward_info=reward_info,
        )
        if not torch.is_tensor(penalty_tensor):
            penalty_tensor = torch.tensor(
                penalty_tensor, dtype=torch.float32, device=base_reward_tensor.device
            )
        penalty_tensor = penalty_tensor.to(device=base_reward_tensor.device, dtype=torch.float32)
        final_reward = base_reward_tensor.to(dtype=torch.float32) * penalty_tensor

        return final_reward, {
            "type": "penalized_reward",
            "base_reward_name": self.base_reward_name,
            "penalty_name": self.penalty_name,
            "base_reward": reward_info,
            "penalty": penalty_info,
            "base_reward_tensor": base_reward_tensor.detach(),
            "penalty_tensor": penalty_tensor.detach(),
            "final_reward_tensor": final_reward.detach(),
        }

import torch

from my_affectgpt.common.registry import registry


class CompositeReward:
    def __init__(self, reward_items):
        self.reward_items = reward_items

    @classmethod
    def from_config(cls, cfg):
        reward_items = []
        for reward_cfg in cfg.get("items", []):
            reward_name = reward_cfg["name"]
            reward_cls = registry.get_reward_class(reward_name)
            assert reward_cls is not None, f"Reward {reward_name} is not registered."
            reward_kwargs = {k: v for k, v in reward_cfg.items() if k != "name"}
            reward_items.append((reward_name, reward_cls.from_config(reward_kwargs)))
        return cls(reward_items)

    def score(self, samples, responses, reward_metas):
        device = None
        total_reward = None
        reward_breakdown = {}

        for reward_name, reward in self.reward_items:
            reward_tensor, reward_info = reward.score(samples, responses, reward_metas)
            if not torch.is_tensor(reward_tensor):
                reward_tensor = torch.tensor(reward_tensor, dtype=torch.float32)
            if device is None:
                device = reward_tensor.device
                total_reward = torch.zeros_like(reward_tensor, dtype=torch.float32, device=device)
            reward_tensor = reward_tensor.to(device=device, dtype=torch.float32)
            total_reward = total_reward + reward.weight * reward_tensor
            reward_breakdown[reward_name] = {
                "weight": reward.weight,
                "reward": reward_tensor.detach(),
                "info": reward_info,
            }

        if total_reward is None:
            raise ValueError("CompositeReward requires at least one reward item.")

        return total_reward, reward_breakdown

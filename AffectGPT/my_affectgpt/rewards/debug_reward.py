import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.rewards.base_reward import BaseReward


@registry.register_reward("debug_constant_reward")
class DebugConstantReward(BaseReward):
    def __init__(self, value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)

    def score(self, samples, responses, reward_metas):
        batch_size = len(responses)
        group_size = len(responses[0]) if batch_size > 0 else 0
        reward_tensor = torch.full((batch_size, group_size), self.value, dtype=torch.float32)
        return reward_tensor, {"type": "constant", "value": self.value}


@registry.register_reward("debug_length_reward")
class DebugLengthReward(BaseReward):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def score(self, samples, responses, reward_metas):
        values = []
        for group in responses:
            values.append([len(response.strip()) * self.scale for response in group])
        reward_tensor = torch.tensor(values, dtype=torch.float32)
        return reward_tensor, {"type": "length", "scale": self.scale}

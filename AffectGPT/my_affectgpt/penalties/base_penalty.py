import torch
import torch.nn as nn


class BasePenalty(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or {}
        if hasattr(cfg, "items"):
            cfg = dict(cfg.items())
        return cls(**cfg)

    def compute(self, samples, responses, reward_metas, reward_tensor, reward_info):
        raise NotImplementedError

    def forward(self, samples, responses, reward_metas, reward_tensor, reward_info):
        return self.compute(samples, responses, reward_metas, reward_tensor, reward_info)

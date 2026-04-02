import torch
import torch.nn as nn


class BaseReward(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = float(weight)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or {}
        if hasattr(cfg, "items"):
            cfg = dict(cfg.items())
        return cls(**cfg)

    def score(self, samples, responses, reward_metas):
        raise NotImplementedError

    def forward(self, samples, responses, reward_metas):
        return self.score(samples, responses, reward_metas)

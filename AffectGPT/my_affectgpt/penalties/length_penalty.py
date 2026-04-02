import math

import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.penalties.base_penalty import BasePenalty


@registry.register_penalty("openset_length_penalty")
class OpenSetLengthPenalty(BasePenalty):
    def __init__(self, penalty_type="p3", deduplicate=True, **kwargs):
        super().__init__(**kwargs)
        self.penalty_type = str(penalty_type).lower()
        self.deduplicate = bool(deduplicate)
        if self.penalty_type not in {"p1", "p2", "p3"}:
            raise ValueError(f"Unsupported penalty_type: {penalty_type}")

    def _normalize_count(self, labels):
        if labels is None:
            return 0
        values = []
        for label in labels:
            label = str(label).strip().lower()
            if label:
                values.append(label)
        if self.deduplicate:
            values = list(dict.fromkeys(values))
        return len(values)

    def _infer_lengths_from_lists(self, reward_info, pred_key, gt_key):
        pred_lists = reward_info.get(pred_key)
        gt_lists = reward_info.get(gt_key)
        if pred_lists is None or gt_lists is None:
            raise ValueError(
                "Length penalty requires pred_lengths/gt_lengths or pred_openset_lists/gt_openset_lists."
            )

        pred_lengths = []
        gt_lengths = []
        for pred_group, gt_group in zip(pred_lists, gt_lists):
            pred_lengths.append([self._normalize_count(labels) for labels in pred_group])
            gt_lengths.append([self._normalize_count(labels) for labels in gt_group])
        return pred_lengths, gt_lengths

    def _resolve_lengths(self, reward_info):
        pred_lengths = reward_info.get("pred_lengths")
        gt_lengths = reward_info.get("gt_lengths")
        if pred_lengths is not None and gt_lengths is not None:
            return pred_lengths, gt_lengths
        return self._infer_lengths_from_lists(
            reward_info, "pred_openset_lists", "gt_openset_lists"
        )

    def _compute_penalty_value(self, pred_length, gt_length):
        if gt_length <= 0:
            raise ValueError("Ground-truth length must be positive for length penalty.")
        if pred_length <= 0 or pred_length <= gt_length:
            return 1.0

        if self.penalty_type == "p1":
            return 1.0 / (pred_length - gt_length + 1.0)
        if self.penalty_type == "p2":
            return gt_length / pred_length
        return math.log(gt_length) / math.log(pred_length)

    def compute(self, samples, responses, reward_metas, reward_tensor, reward_info):
        pred_lengths, gt_lengths = self._resolve_lengths(reward_info)
        penalty_values = []
        for pred_group, gt_group in zip(pred_lengths, gt_lengths):
            group_penalty = []
            for pred_length, gt_length in zip(pred_group, gt_group):
                group_penalty.append(self._compute_penalty_value(pred_length, gt_length))
            penalty_values.append(group_penalty)

        device = reward_tensor.device if torch.is_tensor(reward_tensor) else None
        penalty_tensor = torch.tensor(penalty_values, dtype=torch.float32, device=device)
        penalty_info = {
            "type": "openset_length_penalty",
            "penalty_type": self.penalty_type,
            "deduplicate": self.deduplicate,
            "pred_lengths": pred_lengths,
            "gt_lengths": gt_lengths,
            "penalty_tensor": penalty_tensor.detach(),
        }
        return penalty_tensor, penalty_info

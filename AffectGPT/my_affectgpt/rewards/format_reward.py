import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.rewards.base_reward import BaseReward


@registry.register_reward("format_reward")
class FormatReward(BaseReward):
    def __init__(
        self,
        think_open_tag="<think>",
        think_close_tag="</think>",
        answer_open_tag="<answer>",
        answer_close_tag="</answer>",
        require_non_empty=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag
        self.require_non_empty = bool(require_non_empty)

    def _extract_block(self, text, open_tag, close_tag):
        start = text.find(open_tag)
        if start < 0:
            return None, -1, -1

        content_start = start + len(open_tag)
        end = text.find(close_tag, content_start)
        if end < 0:
            return None, start, -1

        return text[content_start:end], start, end + len(close_tag)

    def _is_required_format(self, text):
        if not isinstance(text, str):
            return False

        think_content, think_start, think_end = self._extract_block(
            text, self.think_open_tag, self.think_close_tag
        )
        answer_content, answer_start, answer_end = self._extract_block(
            text, self.answer_open_tag, self.answer_close_tag
        )

        if think_content is None or answer_content is None:
            return False

        if not (0 <= think_start < think_end <= answer_start < answer_end):
            return False

        if self.require_non_empty:
            if not think_content.strip() or not answer_content.strip():
                return False

        return True

    def score(self, samples, responses, reward_metas):
        values = []
        for group in responses:
            values.append([1.0 if self._is_required_format(response) else 0.0 for response in group])

        reward_tensor = torch.tensor(values, dtype=torch.float32)
        return reward_tensor, {
            "type": "format",
            "think_open_tag": self.think_open_tag,
            "think_close_tag": self.think_close_tag,
            "answer_open_tag": self.answer_open_tag,
            "answer_close_tag": self.answer_close_tag,
            "require_non_empty": self.require_non_empty,
        }

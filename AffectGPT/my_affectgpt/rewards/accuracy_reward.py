import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.evaluation.ew_metric import extract_openset_texts_batch, func_read_batch_calling_model
from my_affectgpt.evaluation.wheel import compute_single_ew_scores
from my_affectgpt.rewards.base_reward import BaseReward
from my_affectgpt.rewards.text_extract_utils import (
    extract_tagged_block,
    log_extractor_warning,
    reshape_like_groups,
    should_reraise_extractor_error,
    unique_length,
)
from toolkit.utils.functions import string_to_list


@registry.register_reward("accuracy_reward")
class AccuracyReward(BaseReward):
    def __init__(
        self,
        extractor_model="Qwen25",
        extractor_batch_size=8,
        strict_answer_tag=True,
        zero_on_extract_failure=True,
        inter_print=False,
        answer_open_tag="<answer>",
        answer_close_tag="</answer>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extractor_model = extractor_model
        self.extractor_batch_size = int(extractor_batch_size)
        self.strict_answer_tag = bool(strict_answer_tag)
        self.zero_on_extract_failure = bool(zero_on_extract_failure)
        self.inter_print = bool(inter_print)
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag

        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

    def _ensure_extractor(self):
        if self._llm is None or self._tokenizer is None or self._sampling_params is None:
            self._llm, self._tokenizer, self._sampling_params = func_read_batch_calling_model(
                self.extractor_model
            )

    def _extract_answer_text(self, response):
        return extract_tagged_block(
            response,
            self.answer_open_tag,
            self.answer_close_tag,
            strict=self.strict_answer_tag,
        )

    def _reshape_like_groups(self, flat_items, group_sizes):
        return reshape_like_groups(flat_items, group_sizes, caller_name="AccuracyReward")

    def _extract_gt_labels(self, reward_meta):
        gt_labels = reward_meta.get("gt_openset_list")
        if gt_labels is None:
            gt_labels = string_to_list(reward_meta.get("gt_openset_text", ""))
        elif isinstance(gt_labels, str):
            gt_labels = string_to_list(gt_labels)
        gt_labels = [item.lower().strip() for item in gt_labels if str(item).strip()]
        if len(gt_labels) == 0:
            raise ValueError("Ground-truth openset labels are empty in reward_meta.")
        return gt_labels

    def _unique_length(self, labels):
        return unique_length(labels)

    def score(self, samples, responses, reward_metas):
        group_sizes = [len(group) for group in responses]
        flat_responses = [response for group in responses for response in group]
        flat_answer_texts = [self._extract_answer_text(response) for response in flat_responses]
        flat_valid_mask = [answer_text is not None for answer_text in flat_answer_texts]

        valid_answer_texts = [answer_text for answer_text in flat_answer_texts if answer_text is not None]
        extracted_openset_texts = []
        if len(valid_answer_texts) > 0:
            try:
                self._ensure_extractor()
                extracted_openset_texts = extract_openset_texts_batch(
                    valid_answer_texts,
                    llm=self._llm,
                    tokenizer=self._tokenizer,
                    sampling_params=self._sampling_params,
                    batch_size=self.extractor_batch_size,
                )
            except Exception as error:
                if should_reraise_extractor_error(error):
                    raise
                log_extractor_warning("AccuracyReward", len(valid_answer_texts), error)
                if not self.zero_on_extract_failure:
                    raise
                extracted_openset_texts = [""] * len(valid_answer_texts)
        # 抽取情感标签

        flat_pred_texts = []
        pred_cursor = 0
        for is_valid in flat_valid_mask:
            if is_valid:
                flat_pred_texts.append(extracted_openset_texts[pred_cursor])
                pred_cursor += 1
            else:
                flat_pred_texts.append("")
        # 将所有有效的情感标签拼接成一个列表

        flat_pred_labels = [
            [item.lower().strip() for item in string_to_list(pred_text) if str(item).strip()]
            for pred_text in flat_pred_texts
        ]
        

        flat_gt_labels = []
        for reward_meta, group_size in zip(reward_metas, group_sizes):
            gt_labels = self._extract_gt_labels(reward_meta)
            flat_gt_labels.extend([list(gt_labels) for _ in range(group_size)])

        flat_precision = []
        flat_recall = []
        flat_f1 = []
        flat_pred_lengths = []
        flat_gt_lengths = []
        for pred_labels, gt_labels in zip(flat_pred_labels, flat_gt_labels):
            score_dict = compute_single_ew_scores(gt_labels, pred_labels)
            flat_f1.append(score_dict["f1"])
            flat_precision.append(score_dict["precision"])
            flat_recall.append(score_dict["recall"])
            flat_pred_lengths.append(self._unique_length(pred_labels))
            flat_gt_lengths.append(self._unique_length(gt_labels))

        reward_tensor = torch.tensor(
            self._reshape_like_groups(flat_f1, group_sizes), dtype=torch.float32
        )

        reward_info = {
            "type": "accuracy",
            "score_type": "f1",
            "valid_answer_mask": self._reshape_like_groups(flat_valid_mask, group_sizes),
            "answer_texts": self._reshape_like_groups(flat_answer_texts, group_sizes),
            "pred_openset_texts": self._reshape_like_groups(flat_pred_texts, group_sizes),
            "pred_openset_lists": self._reshape_like_groups(flat_pred_labels, group_sizes),
            "gt_openset_lists": self._reshape_like_groups(flat_gt_labels, group_sizes),
            "pred_lengths": self._reshape_like_groups(flat_pred_lengths, group_sizes),
            "gt_lengths": self._reshape_like_groups(flat_gt_lengths, group_sizes),
            "precision": self._reshape_like_groups(flat_precision, group_sizes),
            "recall": self._reshape_like_groups(flat_recall, group_sizes),
            "f1": self._reshape_like_groups(flat_f1, group_sizes),
            "extractor_model": self.extractor_model,
        }
        return reward_tensor, reward_info

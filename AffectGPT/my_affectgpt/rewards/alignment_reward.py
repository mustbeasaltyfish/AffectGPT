import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.evaluation.ew_metric import extract_openset_texts_batch, func_read_batch_calling_model
from my_affectgpt.evaluation.wheel import compute_single_ew_jaccard_scores
from my_affectgpt.rewards.base_reward import BaseReward
from my_affectgpt.rewards.text_extract_utils import (
    extract_tagged_block,
    log_extractor_warning,
    reshape_like_groups,
    should_reraise_extractor_error,
    unique_length,
)
from toolkit.utils.functions import string_to_list


@registry.register_reward("alignment_reward")
class AlignmentReward(BaseReward):
    def __init__(
        self,
        extractor_model="Qwen25",
        extractor_batch_size=8,
        strict_think_tag=True,
        strict_answer_tag=True,
        zero_on_extract_failure=True,
        inter_print=False,
        think_open_tag="<think>",
        think_close_tag="</think>",
        answer_open_tag="<answer>",
        answer_close_tag="</answer>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extractor_model = extractor_model
        self.extractor_batch_size = int(extractor_batch_size)
        self.strict_think_tag = bool(strict_think_tag)
        self.strict_answer_tag = bool(strict_answer_tag)
        self.zero_on_extract_failure = bool(zero_on_extract_failure)
        self.inter_print = bool(inter_print)
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag
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

    def _extract_think_text(self, response):
        return extract_tagged_block(
            response,
            self.think_open_tag,
            self.think_close_tag,
            strict=self.strict_think_tag,
        )

    def _extract_answer_text(self, response):
        return extract_tagged_block(
            response,
            self.answer_open_tag,
            self.answer_close_tag,
            strict=self.strict_answer_tag,
        )

    def _reshape_like_groups(self, flat_items, group_sizes):
        return reshape_like_groups(flat_items, group_sizes, caller_name="AlignmentReward")

    def _unique_length(self, labels):
        return unique_length(labels)

    def _normalize_extracted_labels(self, extracted_text):
        return [item.lower().strip() for item in string_to_list(extracted_text) if str(item).strip()]

    def _extract_openset_texts(self, texts, side_name):
        if len(texts) == 0:
            return []
        try:
            self._ensure_extractor()
            return extract_openset_texts_batch(
                texts,
                llm=self._llm,
                tokenizer=self._tokenizer,
                sampling_params=self._sampling_params,
                batch_size=self.extractor_batch_size,
            )
        except Exception as error:
            if should_reraise_extractor_error(error):
                raise
            log_extractor_warning(f"AlignmentReward[{side_name}]", len(texts), error)
            if not self.zero_on_extract_failure:
                raise
            return [""] * len(texts)

    def score(self, samples, responses, reward_metas):
        del reward_metas

        group_sizes = [len(group) for group in responses]
        flat_responses = [response for group in responses for response in group]

        flat_think_texts = [self._extract_think_text(response) for response in flat_responses]
        flat_answer_texts = [self._extract_answer_text(response) for response in flat_responses]

        flat_valid_think_mask = [think_text is not None for think_text in flat_think_texts]
        flat_valid_answer_mask = [answer_text is not None for answer_text in flat_answer_texts]
        flat_valid_pair_mask = [
            think_valid and answer_valid
            for think_valid, answer_valid in zip(flat_valid_think_mask, flat_valid_answer_mask)
        ]

        valid_think_texts = [think_text for think_text in flat_think_texts if think_text is not None]
        valid_answer_texts = [answer_text for answer_text in flat_answer_texts if answer_text is not None]

        extracted_think_texts = self._extract_openset_texts(valid_think_texts, side_name="think")
        extracted_answer_texts = self._extract_openset_texts(valid_answer_texts, side_name="answer")

        flat_think_pred_texts = []
        think_cursor = 0
        for is_valid in flat_valid_think_mask:
            if is_valid:
                flat_think_pred_texts.append(extracted_think_texts[think_cursor])
                think_cursor += 1
            else:
                flat_think_pred_texts.append("")

        flat_answer_pred_texts = []
        answer_cursor = 0
        for is_valid in flat_valid_answer_mask:
            if is_valid:
                flat_answer_pred_texts.append(extracted_answer_texts[answer_cursor])
                answer_cursor += 1
            else:
                flat_answer_pred_texts.append("")

        flat_think_labels = [
            self._normalize_extracted_labels(pred_text) for pred_text in flat_think_pred_texts
        ]
        flat_answer_labels = [
            self._normalize_extracted_labels(pred_text) for pred_text in flat_answer_pred_texts
        ]

        flat_similarity = []
        flat_think_lengths = []
        flat_answer_lengths = []
        for is_valid_pair, think_labels, answer_labels in zip(
            flat_valid_pair_mask, flat_think_labels, flat_answer_labels
        ):
            if not is_valid_pair:
                similarity = 0.0
            else:
                score_dict = compute_single_ew_jaccard_scores(think_labels, answer_labels)
                similarity = score_dict["similarity"]
            flat_similarity.append(similarity)
            flat_think_lengths.append(self._unique_length(think_labels))
            flat_answer_lengths.append(self._unique_length(answer_labels))

        reward_tensor = torch.tensor(
            self._reshape_like_groups(flat_similarity, group_sizes), dtype=torch.float32
        )

        reward_info = {
            "type": "alignment",
            "score_type": "ew_jaccard",
            "valid_think_mask": self._reshape_like_groups(flat_valid_think_mask, group_sizes),
            "valid_answer_mask": self._reshape_like_groups(flat_valid_answer_mask, group_sizes),
            "think_texts": self._reshape_like_groups(flat_think_texts, group_sizes),
            "answer_texts": self._reshape_like_groups(flat_answer_texts, group_sizes),
            "think_openset_texts": self._reshape_like_groups(flat_think_pred_texts, group_sizes),
            "answer_openset_texts": self._reshape_like_groups(flat_answer_pred_texts, group_sizes),
            "think_openset_lists": self._reshape_like_groups(flat_think_labels, group_sizes),
            "answer_openset_lists": self._reshape_like_groups(flat_answer_labels, group_sizes),
            "think_lengths": self._reshape_like_groups(flat_think_lengths, group_sizes),
            "answer_lengths": self._reshape_like_groups(flat_answer_lengths, group_sizes),
            "similarity": self._reshape_like_groups(flat_similarity, group_sizes),
            "extractor_model": self.extractor_model,
        }
        return reward_tensor, reward_info

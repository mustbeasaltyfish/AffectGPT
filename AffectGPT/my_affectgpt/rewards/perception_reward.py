import logging
import math

import numpy as np
import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.rewards.base_reward import BaseReward
from my_affectgpt.rewards.perception_judge import GemmaPerceptionJudge
from my_affectgpt.rewards.text_extract_utils import extract_tagged_block, reshape_like_groups


@registry.register_reward("perception_reward")
class PerceptionReward(BaseReward):
    def __init__(
        self,
        judge_model_id="google/gemma-4-E4B-it",
        judge_dtype="bfloat16",
        judge_device_map="auto",
        max_video_seconds=60,
        max_audio_seconds=30,
        video_fps=1.0,
        strict_answer_tag=True,
        zero_on_judge_failure=True,
        batch_pair_requests=False,
        inter_print=False,
        answer_open_tag="<answer>",
        answer_close_tag="</answer>",
        judge=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model_id = judge_model_id
        self.judge_dtype = judge_dtype
        self.judge_device_map = judge_device_map
        self.max_video_seconds = int(max_video_seconds)
        self.max_audio_seconds = int(max_audio_seconds)
        self.video_fps = float(video_fps)
        self.strict_answer_tag = bool(strict_answer_tag)
        self.zero_on_judge_failure = bool(zero_on_judge_failure)
        self.batch_pair_requests = bool(batch_pair_requests)
        self.inter_print = bool(inter_print)
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag
        self.audio_sampling_rate = 16000

        self._judge = judge

    def _ensure_judge(self):
        if self._judge is None:
            self._judge = GemmaPerceptionJudge(
                model_id=self.judge_model_id,
                dtype=self.judge_dtype,
                device_map=self.judge_device_map,
            )

    def _extract_answer_text(self, response):
        return extract_tagged_block(
            response,
            self.answer_open_tag,
            self.answer_close_tag,
            strict=self.strict_answer_tag,
        )

    def _reshape_like_groups(self, flat_items, group_sizes):
        return reshape_like_groups(flat_items, group_sizes, caller_name="PerceptionReward")

    def _uniform_indices(self, total, target):
        if total <= target:
            return list(range(total))
        return np.linspace(0, total - 1, num=target, dtype=int).tolist()

    def _get_visual_tensor(self, samples, sample_idx):
        if "raw_frames" in samples:
            return samples["raw_frames"][sample_idx], "raw_frames"
        if "raw_faces" in samples:
            return samples["raw_faces"][sample_idx], "raw_faces"
        raise ValueError("PerceptionReward requires raw_frames or raw_faces in the batch.")

    def _prepare_video_array(self, samples, sample_idx):
        raw_video, source_name = self._get_visual_tensor(samples, sample_idx)
        video_tensor = raw_video.detach().cpu()
        if video_tensor.ndim != 4:
            raise ValueError(f"Unexpected raw video shape: {tuple(video_tensor.shape)}")

        if video_tensor.shape[0] in {1, 3}:
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        elif video_tensor.shape[-1] not in {1, 3}:
            raise ValueError(f"Unexpected raw video channel layout: {tuple(video_tensor.shape)}")

        max_frames = max(1, int(self.max_video_seconds * self.video_fps))
        frame_indices = self._uniform_indices(video_tensor.shape[0], max_frames)
        video_tensor = video_tensor[frame_indices]
        video_array = video_tensor.numpy()
        if video_array.dtype != np.uint8:
            video_array = np.clip(video_array, 0, 255).astype(np.uint8)
        return video_array, source_name

    def _prepare_audio_array(self, samples, sample_idx):
        if "raw_audios" not in samples:
            raise ValueError("PerceptionReward requires raw_audios in the batch.")

        audio_tensor = samples["raw_audios"][sample_idx].detach().cpu()
        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(1)
        elif audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim != 2:
            raise ValueError(f"Unexpected raw audio shape: {tuple(audio_tensor.shape)}")

        waveform = audio_tensor.reshape(1, -1)
        max_samples = self.audio_sampling_rate * max(1, self.max_audio_seconds)
        waveform = waveform[:, :max_samples]
        return waveform.numpy().astype(np.float32)

    def _judge_pairs(self, pair_inputs):
        self._ensure_judge()
        if self.batch_pair_requests:
            return self._judge.compare_pairs(
                pair_inputs,
                audio_sampling_rate=self.audio_sampling_rate,
            )
        verdicts = []
        for pair_input in pair_inputs:
            verdicts.append(
                self._judge.compare_pair(
                    video_array=pair_input["video_array"],
                    audio_array=pair_input["audio_array"],
                    subtitle=pair_input["subtitle"],
                    answer_a=pair_input["answer_a"],
                    answer_b=pair_input["answer_b"],
                    audio_sampling_rate=self.audio_sampling_rate,
                )
            )
        return verdicts

    def _fit_bradley_terry(self, group_size, winner_loser_pairs, max_iter=200, tol=1e-8, eps=1e-8):
        if len(winner_loser_pairs) == 0:
            return None

        wins = np.zeros(group_size, dtype=np.float64)
        comparisons = np.zeros((group_size, group_size), dtype=np.float64)
        for winner, loser in winner_loser_pairs:
            wins[winner] += 1.0
            comparisons[winner, loser] += 1.0
            comparisons[loser, winner] += 1.0

        skills = np.ones(group_size, dtype=np.float64)
        for _ in range(max_iter):
            previous = skills.copy()
            updated = np.zeros_like(skills)
            for i in range(group_size):
                denom = 0.0
                for j in range(group_size):
                    if i == j or comparisons[i, j] <= 0:
                        continue
                    denom += comparisons[i, j] / max(skills[i] + skills[j], eps)
                if denom > 0:
                    updated[i] = (wins[i] + eps) / max(denom, eps)
                else:
                    updated[i] = eps

            updated = np.maximum(updated, eps)
            updated /= updated.mean()
            skills = updated
            if np.max(np.abs(skills - previous)) < tol:
                break

        return np.log(skills + eps)

    def _compute_group_reward(self, group_size, bt_scores):
        top_k = int(math.ceil(group_size / 2))
        rank_indices = np.argsort(-bt_scores, kind="mergesort").tolist()
        top_indices = set(rank_indices[:top_k])
        reward_binary = [1.0 if index in top_indices else 0.0 for index in range(group_size)]
        return reward_binary, rank_indices, top_k

    def score(self, samples, responses, reward_metas):
        group_sizes = [len(group) for group in responses]
        flat_reward = []
        group_answer_texts = []
        group_valid_answer_masks = []
        group_pair_indices = []
        group_pair_verdicts = []
        group_pair_success_masks = []
        group_bt_scores = []
        group_rank_indices = []
        group_top_k = []
        group_reward_binary = []
        group_visual_sources = []

        for sample_idx, (group_responses, reward_meta) in enumerate(zip(responses, reward_metas)):
            group_size = len(group_responses)
            answer_texts = [self._extract_answer_text(response) for response in group_responses]
            valid_answer_mask = [answer_text is not None for answer_text in answer_texts]

            if group_size < 2:
                reward_binary = [1.0] * group_size
                bt_scores = [0.0] * group_size
                rank_indices = list(range(group_size))
                top_k = group_size
                pair_indices = []
                pair_verdicts = []
                pair_success_masks = []
                visual_source = None
            else:
                try:
                    video_array, visual_source = self._prepare_video_array(samples, sample_idx)
                    audio_array = self._prepare_audio_array(samples, sample_idx)
                except Exception:
                    if not self.zero_on_judge_failure:
                        raise
                    logging.warning(
                        "PerceptionReward failed to prepare judge inputs for sample %s.",
                        reward_meta.get("name", sample_idx),
                    )
                    reward_binary = [0.0] * group_size
                    bt_scores = [0.0] * group_size
                    rank_indices = list(range(group_size))
                    top_k = int(math.ceil(group_size / 2))
                    pair_indices = []
                    pair_verdicts = []
                    pair_success_masks = []
                    visual_source = None
                else:
                    pair_indices = []
                    pair_verdicts = []
                    pair_success_masks = []
                    winner_loser_pairs = []
                    judge_batch = []
                    judge_batch_meta = []

                    for i in range(group_size):
                        for j in range(i + 1, group_size):
                            pair_indices.append((i, j))
                            answer_i = answer_texts[i]
                            answer_j = answer_texts[j]
                            valid_i = answer_i is not None
                            valid_j = answer_j is not None

                            if valid_i and not valid_j:
                                pair_verdicts.append("A")
                                pair_success_masks.append(True)
                                winner_loser_pairs.append((i, j))
                                continue
                            if not valid_i and valid_j:
                                pair_verdicts.append("B")
                                pair_success_masks.append(True)
                                winner_loser_pairs.append((j, i))
                                continue
                            if not valid_i and not valid_j:
                                pair_verdicts.append(None)
                                pair_success_masks.append(False)
                                continue

                            pair_verdicts.append(None)
                            pair_success_masks.append(False)
                            judge_batch.append(
                                {
                                    "video_array": video_array,
                                    "audio_array": audio_array,
                                    "subtitle": reward_meta.get("subtitle", ""),
                                    "answer_a": answer_i,
                                    "answer_b": answer_j,
                                }
                            )
                            judge_batch_meta.append(len(pair_indices) - 1)

                    if len(judge_batch) > 0:
                        try:
                            judge_verdicts = self._judge_pairs(judge_batch)
                        except Exception as error:
                            if not self.zero_on_judge_failure:
                                raise
                            logging.warning(
                                "PerceptionReward judge failed for sample %s with %s: %s",
                                reward_meta.get("name", sample_idx),
                                type(error).__name__,
                                error,
                            )
                            judge_verdicts = [None] * len(judge_batch)

                        for meta_index, verdict in zip(judge_batch_meta, judge_verdicts):
                            if verdict not in {"A", "B"}:
                                continue
                            pair_verdicts[meta_index] = verdict
                            pair_success_masks[meta_index] = True
                            pair_i, pair_j = pair_indices[meta_index]
                            if verdict == "A":
                                winner_loser_pairs.append((pair_i, pair_j))
                            else:
                                winner_loser_pairs.append((pair_j, pair_i))

                    bt_scores_array = self._fit_bradley_terry(group_size, winner_loser_pairs)
                    if bt_scores_array is None:
                        reward_binary = [0.0] * group_size
                        bt_scores = [0.0] * group_size
                        rank_indices = list(range(group_size))
                        top_k = int(math.ceil(group_size / 2))
                    else:
                        bt_scores = bt_scores_array.tolist()
                        reward_binary, rank_indices, top_k = self._compute_group_reward(
                            group_size, bt_scores_array
                        )

            flat_reward.extend(reward_binary)
            group_answer_texts.append(answer_texts)
            group_valid_answer_masks.append(valid_answer_mask)
            group_pair_indices.append(pair_indices)
            group_pair_verdicts.append(pair_verdicts)
            group_pair_success_masks.append(pair_success_masks)
            group_bt_scores.append(bt_scores)
            group_rank_indices.append(rank_indices)
            group_top_k.append(top_k)
            group_reward_binary.append(reward_binary)
            group_visual_sources.append(visual_source)

        reward_tensor = torch.tensor(
            self._reshape_like_groups(flat_reward, group_sizes), dtype=torch.float32
        )

        reward_info = {
            "type": "perception",
            "score_type": "top50_bt",
            "group_size": group_sizes,
            "answer_texts": group_answer_texts,
            "valid_answer_mask": group_valid_answer_masks,
            "pair_indices": group_pair_indices,
            "pair_verdicts": group_pair_verdicts,
            "pair_success_mask": group_pair_success_masks,
            "bt_scores": group_bt_scores,
            "rank_indices": group_rank_indices,
            "top_k": group_top_k,
            "reward_binary": group_reward_binary,
            "judge_model_id": self.judge_model_id,
            "visual_source": group_visual_sources,
        }
        return reward_tensor, reward_info

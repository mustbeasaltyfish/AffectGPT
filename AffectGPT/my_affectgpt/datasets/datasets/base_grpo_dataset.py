import copy
import random
import warnings

import torch

from my_affectgpt.datasets.datasets.base_dataset import BaseDataset


class BaseGRPODataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_visualize_samples(self, sample):
        print(sample["prompt_text"])
        print(sample["reward_meta"])

    def build_prompt_text(self, sample):
        raise NotImplementedError

    def build_reward_meta(self, sample):
        raise NotImplementedError

    def __getitem__(self, index):
        num_retries = 10
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                video_path, image_path, audio_path, face_npy = None, None, None, None
                if hasattr(self, "_get_video_path"):
                    video_path = self._get_video_path(sample)
                if hasattr(self, "_get_image_path"):
                    image_path = self._get_image_path(sample)
                if hasattr(self, "_get_audio_path"):
                    audio_path = self._get_audio_path(sample)
                if hasattr(self, "_get_face_path"):
                    face_npy = self._get_face_path(sample)

                sample_data = self.read_frame_face_audio_text(
                    video_path, face_npy, audio_path, image_path
                )
                prompt_text = self.build_prompt_text(sample)
                reward_meta = self.build_reward_meta(sample)
            except Exception as error:
                print(f"Error: {error}")
                print(
                    f"Failed to load data {self.dataset} {sample.get('name', 'unknown')}. "
                    "We will randomly sample an example as a replacement."
                )
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch sample after {num_retries} retries.")

        return {
            "sample_id": sample["name"],
            "face": sample_data["face"],
            "raw_face": sample_data["raw_face"],
            "frame": sample_data["frame"],
            "raw_frame": sample_data["raw_frame"],
            "audio": sample_data["audio"],
            "raw_audio": sample_data["raw_audio"],
            "image": sample_data["image"],
            "raw_image": sample_data["raw_image"],
            "prompt_text": prompt_text,
            "question_type": sample.get("question_type", "ovlabel"),
            "reward_meta": reward_meta,
            "dataset": self.dataset.lower(),
            "face_or_frame": self.face_or_frame,
        }

    def collater(self, instances):
        batch = {
            "sample_ids": [instance["sample_id"] for instance in instances],
            "prompt_texts": [instance["prompt_text"] for instance in instances],
            "question_types": [instance["question_type"] for instance in instances],
            "reward_metas": [copy.deepcopy(instance["reward_meta"]) for instance in instances],
            "dataset": instances[0]["dataset"],
            "face_or_frame": instances[0]["face_or_frame"],
        }

        for sample_type in [
            "face",
            "raw_face",
            "frame",
            "raw_frame",
            "audio",
            "raw_audio",
            "image",
            "raw_image",
        ]:
            batch_type = sample_type + "s"
            datas = [instance[sample_type] for instance in instances]
            if all(x is None for x in datas):
                continue
            if any(x is None for x in datas):
                raise ValueError(f"Inconsistent None values detected in GRPO batch for {sample_type}.")
            if all(x.shape == datas[0].shape for x in datas):
                batch[batch_type] = torch.stack(datas)
            else:
                warnings.warn(
                    f"Shape mismatch detected in GRPO batch for {sample_type}: "
                    f"{[tuple(x.shape) for x in datas]}",
                    stacklevel=2,
                )
                raise ValueError(f"Shape mismatch detected in GRPO batch for {sample_type}.")

        return batch

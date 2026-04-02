import logging

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from my_affectgpt.datasets.datasets.mer2025ov_dataset import MER2025OV_GRPO_Dataset


@registry.register_builder("mer2025ov_grpo")
class MER2025OVGRPOBuilder(BaseDatasetBuilder):
    train_dataset_cls = MER2025OV_GRPO_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2025OV_GRPO_Dataset")
        self.build_processors()

        datasets = dict()
        datasets["train"] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
        )
        return datasets

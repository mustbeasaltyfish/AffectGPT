import json
import logging
import os
from contextlib import nullcontext

import torch

from my_affectgpt.common.dist_utils import main_process
from my_affectgpt.common.logger import MetricLogger, SmoothedValue
from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.data_utils import prepare_sample
from my_affectgpt.runners.runner_base import RunnerBase


@registry.register_runner("runner_grpo")
class GRPORunner(RunnerBase):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg=cfg, task=task, model=model, datasets=datasets, job_id=job_id)
        self._ref_model = None
        self._rewarder = None

    @property
    def ref_model(self):
        if self._ref_model is None:
            self._ref_model = self.task.build_ref_model(self.config).to(self.device)
        return self._ref_model

    @property
    def rewarder(self):
        if self._rewarder is None:
            self._rewarder = self.task.build_rewards(self.config)
        return self._rewarder

    def train_epoch(self, epoch):
        actor_model = self.unwrap_dist_model(self.model)
        actor_model.train()
        use_ref_model = float(self.config.grpo_cfg.get("kl_coef", 0.02)) > 0
        if use_ref_model:
            self.ref_model.eval()

        data_loader = self.train_loader
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("reward", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("kl", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("ratio", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("response_len", SmoothedValue(window_size=1, fmt="{value:.8f}"))

        iters_per_epoch = self.lr_scheduler.iters_per_epoch
        use_amp = self.scaler is not None
        header = "GRPO Train: data epoch: [{}]".format(epoch)

        logging.info(
            "Start GRPO training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )

        for i in metric_logger.log_every(range(iters_per_epoch), self.log_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=i)

            rollout_outputs = self.task.rollout_step(actor_model, samples, self.config.grpo_cfg)
            reward_tensor, _ = self.task.compute_rewards(
                self.rewarder, samples, rollout_outputs["responses"]
            )
            advantages = self.task.compute_advantages(reward_tensor, self.config.grpo_cfg)

            autocast_ctx = actor_model.maybe_autocast() if use_amp else nullcontext()
            with autocast_ctx:
                train_outputs = self.task.train_step(
                    actor_model,
                    self.ref_model if use_ref_model else None,
                    samples,
                    rollout_outputs["prompt_input_ids"],
                    rollout_outputs["response_token_ids"],
                    rollout_outputs["old_token_logprobs"],
                    advantages,
                    self.config.grpo_cfg,
                )

            loss = train_outputs["loss"]
            if use_amp:
                self.scaler.scale(loss).backward()
                max_grad_norm = float(self.config.grpo_cfg.get("max_grad_norm", 1.0))
                if max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                max_grad_norm = float(self.config.grpo_cfg.get("max_grad_norm", 1.0))
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(reward=reward_tensor.mean().item())
            metric_logger.update(kl=float(train_outputs["kl_loss"]))
            metric_logger.update(ratio=float(train_outputs["ratio_mean"]))
            metric_logger.update(response_len=float(train_outputs["response_len"]))

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

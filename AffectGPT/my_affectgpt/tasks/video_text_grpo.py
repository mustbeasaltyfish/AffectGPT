from omegaconf import OmegaConf
import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.rewards.composite_reward import CompositeReward
from my_affectgpt.rl.grpo_loss import compute_group_advantages, compute_grpo_loss
from my_affectgpt.tasks.base_task import BaseTask


@registry.register_task("video_text_grpo")
class VideoTextGRPOTask(BaseTask):
    def __init__(self):
        super().__init__()

    def build_ref_model(self, cfg):
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model_cfg, resolve=True))
        actor_ckpt = cfg.grpo_cfg.get("actor_ckpt", "")
        ref_ckpt = cfg.grpo_cfg.get("ref_ckpt", "")
        if actor_ckpt:
            model_cfg.ckpt = actor_ckpt
        if ref_ckpt:
            model_cfg.ckpt_2 = ref_ckpt
        model = self.build_model_from_cfg(model_cfg)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    def build_model_from_cfg(self, model_cfg):
        model_cls = registry.get_model_class(model_cfg.arch)
        return model_cls.from_config(model_cfg)

    def build_rewards(self, cfg):
        return CompositeReward.from_config(cfg.rewards_cfg)

    def rollout_step(self, model, samples, grpo_cfg):
        generation_cfg = {
            "max_new_tokens": int(grpo_cfg.get("max_new_tokens", 256)),
            "temperature": float(grpo_cfg.get("temperature", 0.8)),
            "top_p": float(grpo_cfg.get("top_p", 0.9)),
        }
        rollout_outputs = model.generate_group(
            samples=samples,
            generation_cfg=generation_cfg,
            group_size=int(grpo_cfg.get("group_size", 4)),
        )
        with torch.no_grad():
            old_policy_outputs = model.compute_response_logprobs(
                samples, rollout_outputs["prompt_input_ids"], rollout_outputs["response_token_ids"]
            )
        rollout_outputs["old_token_logprobs"] = old_policy_outputs["token_logprobs"].detach()
        rollout_outputs["old_seq_logprobs"] = old_policy_outputs["seq_logprobs"].detach()
        return rollout_outputs

    def compute_rewards(self, rewarder, samples, responses):
        return rewarder.score(samples, responses, samples["reward_metas"])

    def compute_advantages(self, reward_tensor, grpo_cfg):
        return compute_group_advantages(
            reward_tensor, eps=float(grpo_cfg.get("advantage_eps", 1e-6))
        )

    def train_step(
        self,
        model,
        ref_model,
        samples,
        prompt_input_ids,
        response_token_ids,
        old_token_logprobs,
        advantages,
        grpo_cfg,
    ):
        actor_outputs = model.compute_response_logprobs(samples, prompt_input_ids, response_token_ids)
        ref_token_logprobs = None
        if float(grpo_cfg.get("kl_coef", 0.02)) > 0:
            with torch.no_grad():
                ref_outputs = ref_model.compute_response_logprobs(
                    samples, prompt_input_ids, response_token_ids
                )
            ref_token_logprobs = ref_outputs["token_logprobs"].detach()

        loss_outputs = compute_grpo_loss(
            actor_token_logprobs=actor_outputs["token_logprobs"],
            old_token_logprobs=old_token_logprobs.to(actor_outputs["token_logprobs"].device),
            ref_token_logprobs=ref_token_logprobs,
            response_mask=actor_outputs["response_mask"],
            advantages=advantages.to(actor_outputs["token_logprobs"].device),
            kl_coef=float(grpo_cfg.get("kl_coef", 0.02)),
            clip_range=float(grpo_cfg.get("clip_range", 0.2)),
        )

        response_lengths = actor_outputs["response_mask"].sum(dim=-1).float()
        return {
            "loss": loss_outputs["loss"],
            "policy_loss": loss_outputs["policy_loss"],
            "kl_loss": loss_outputs["kl_loss"],
            "ratio_mean": loss_outputs["ratios"].mean().detach(),
            "response_len": response_lengths.mean().detach(),
        }

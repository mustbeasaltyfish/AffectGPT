import torch


def compute_group_advantages(reward_tensor, eps=1e-6):
    group_mean = reward_tensor.mean(dim=1, keepdim=True)
    group_std = reward_tensor.std(dim=1, keepdim=True, unbiased=False)
    advantages = (reward_tensor - group_mean) / (group_std + eps)
    return advantages


def masked_mean(values, mask, dim=-1):
    mask = mask.to(values.dtype)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (values * mask).sum(dim=dim) / denom


def compute_grpo_loss(
    actor_token_logprobs,
    old_token_logprobs,
    ref_token_logprobs,
    response_mask,
    advantages,
    kl_coef,
    clip_range=0.2,
):
    actor_seq_logprobs = masked_mean(actor_token_logprobs, response_mask, dim=-1)
    old_seq_logprobs = masked_mean(old_token_logprobs, response_mask, dim=-1)
    if ref_token_logprobs is None:
        ref_seq_logprobs = torch.zeros_like(actor_seq_logprobs)
        kl_values = torch.zeros_like(actor_seq_logprobs)
        kl_loss = actor_seq_logprobs.new_tensor(0.0)
    else:
        ref_seq_logprobs = masked_mean(ref_token_logprobs, response_mask, dim=-1)
        kl_values = masked_mean(actor_token_logprobs - ref_token_logprobs, response_mask, dim=-1)
        kl_loss = kl_values.mean()

    ratios = torch.exp(actor_seq_logprobs - old_seq_logprobs)
    unclipped_objective = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
    clipped_objective = clipped_ratios * advantages
    policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
    total_loss = policy_loss + kl_coef * kl_loss

    return {
        "loss": total_loss,
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "actor_seq_logprobs": actor_seq_logprobs.detach(),
        "old_seq_logprobs": old_seq_logprobs.detach(),
        "ref_seq_logprobs": ref_seq_logprobs.detach(),
        "ratios": ratios.detach(),
    }

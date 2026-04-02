import torch


def repeat_interleave_batch(batch_tensor, repeats):
    if batch_tensor is None:
        return None
    return torch.repeat_interleave(batch_tensor, repeats=repeats, dim=0)


def flatten_group_texts(grouped_responses):
    flat_responses = []
    for group in grouped_responses:
        flat_responses.extend(group)
    return flat_responses


def infer_response_lengths(response_token_ids, pad_token_id):
    return response_token_ids.ne(pad_token_id).sum(dim=-1)

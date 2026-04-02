import logging


def extract_tagged_block(text, open_tag, close_tag, strict=True):
    if not isinstance(text, str):
        return None

    start = text.find(open_tag)
    if start < 0:
        return None if strict else text.strip() or None

    content_start = start + len(open_tag)
    end = text.find(close_tag, content_start)
    if end < 0:
        return None if strict else text[content_start:].strip() or None

    content = text[content_start:end].strip()
    return content if content else None


def reshape_like_groups(flat_items, group_sizes, caller_name="reward"):
    if len(set(group_sizes)) > 1:
        raise ValueError(f"{caller_name} requires uniform group sizes, got {group_sizes}.")

    outputs = []
    cursor = 0
    for group_size in group_sizes:
        outputs.append(flat_items[cursor: cursor + group_size])
        cursor += group_size
    return outputs


def unique_length(labels):
    values = []
    for label in labels:
        label = str(label).strip().lower()
        if label:
            values.append(label)
    return len(list(dict.fromkeys(values)))


def should_reraise_extractor_error(error):
    if isinstance(error, (MemoryError, OSError)):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        critical_patterns = [
            "out of memory",
            "cuda",
            "cublas",
            "cudnn",
            "model weights",
            "failed to initialize",
        ]
        return any(pattern in message for pattern in critical_patterns)
    return False


def log_extractor_warning(reward_name, count, error):
    logging.warning(
        "%s extractor failed for %d texts with %s: %s",
        reward_name,
        count,
        type(error).__name__,
        error,
    )

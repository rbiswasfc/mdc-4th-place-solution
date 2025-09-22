from dataclasses import dataclass

import torch


@dataclass
class MDCCollator:
    """
    Data collector for Make Data Count - Finding Data References (MDC).
    This should be used together with flash attention.

    - concatenate the entire mini batch into single long sequence [1, total_tokens]
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, tokenizer, pad_to_multiple_of=32):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        ret = {"input_ids": [], "position_ids": [], "labels": [], "end_idxs": []}

        for feature in features:
            for current_ids in feature["input_ids"]:
                ret["input_ids"] += current_ids
                ret["position_ids"] += list(range(len(current_ids)))
                ret["end_idxs"].append(len(ret["input_ids"]) - 1)
            ret["labels"] += feature["labels"]

        # handle padding (right side, no need to update end_idxs) --
        n_tokens = len(ret["input_ids"])
        n_pad = (
            n_tokens // self.pad_to_multiple_of + 1
        ) * self.pad_to_multiple_of - n_tokens
        last_position_id = ret["position_ids"][-1]

        ret["input_ids"] += [self.pad_token_id] * n_pad
        ret["position_ids"] += list(
            range(last_position_id + 1, last_position_id + n_pad + 1)
        )

        # prepare batch ---
        batch = dict()
        batch["input_ids"] = torch.tensor(
            ret["input_ids"], dtype=torch.int64
        ).unsqueeze(0)
        batch["position_ids"] = torch.tensor(
            ret["position_ids"], dtype=torch.int64
        ).unsqueeze(0)
        batch["end_idxs"] = torch.tensor(ret["end_idxs"], dtype=torch.int64)
        batch["labels"] = torch.tensor(ret["labels"], dtype=torch.float32)

        return batch


def show_batch(batch, tokenizer, print_fn=print, **kwargs):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")
    print_fn(f"shape of position_ids: {batch['position_ids'].shape}")
    print_fn(f"shape of end_idxs: {batch['end_idxs'].shape}")

    if "labels" in batch.keys():
        print_fn(f"shape of labels: {batch['labels'].shape}")
        print_fn(f"labels: {batch['labels']}")

    print_fn("\n\n")
    for idx in range(bs):
        print_fn(f"=== Example {idx} ===")
        print_fn(
            f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}"
        )
        if "labels" in batch.keys():
            print_fn(f"Label: {batch['labels'][idx]}")
        print_fn("~~" * 40)

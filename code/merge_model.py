import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main(args):
    backbone_path = args.backbone_path
    adapter_path = args.adapter_path

    # backbone_path = "Qwen/Qwen2.5-72B" # "Qwen/Qwen2.5-72B-Instruct"
    # adapter_path = "./working/models/qwen72b_classify_fold_3"

    save_dir = os.path.join(adapter_path, "merged")

    config = AutoConfig.from_pretrained(backbone_path, trust_remote_code=False)
    config.use_cache = False

    base_model = AutoModelForCausalLM.from_pretrained(backbone_path, config=config, torch_dtype=torch.bfloat16)

    base_model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload(safe_merge=True)

    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Merged model saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", type=str, default="Qwen/Qwen2.5-72B")
    parser.add_argument("--adapter_path", type=str, default="./working/models/qwen72b_classify_fold_3")
    parser.add_argument("--save_dir", type=str, default="./working/models/qwen72b_classify_fold_3/merged")
    args = parser.parse_args()

    main(args)


# python code/merge_model.py --backbone_path Qwen/Qwen2.5-72B --adapter_path ./working/models/filter_72b_fold_3

# python code/merge_model.py --backbone_path Qwen/Qwen2.5-72B --adapter_path ./working/models/qwen14b_classify_fold_0

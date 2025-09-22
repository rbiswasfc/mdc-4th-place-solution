from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.file_utils import ModelOutput


@dataclass
class MDCModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None


def get_base_model(cfg):
    config = AutoConfig.from_pretrained(
        cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code
    )
    config.use_cache = False

    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    model.config.pretraining_tp = 1

    # LoRA ---
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


class MDCModel(nn.Module):
    def __init__(self, cfg, base_model, tokenizer):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Get token positions once during initialization
        self.register_buffer(
            "yes_loc",
            torch.tensor(tokenizer("Yes", add_special_tokens=False)["input_ids"][-1]),
        )
        self.register_buffer(
            "no_loc",
            torch.tensor(tokenizer("No", add_special_tokens=False)["input_ids"][-1]),
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def get_logits(self, input_ids, position_ids, end_idxs):
        outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        head_logits = outputs.logits.index_select(1, end_idxs)[
            0
        ]  # (num_examples, hidden_size)
        logits = (
            head_logits[:, self.yes_loc] - head_logits[:, self.no_loc]
        ).contiguous()

        return logits

    def forward(self, input_ids, position_ids, end_idxs, labels=None, **kwargs):
        logits = self.get_logits(input_ids, position_ids, end_idxs)

        loss = None

        if labels is not None:
            labels = labels.to(logits.device).reshape(-1)  # bs
            loss = self.loss_fn(logits, labels)

        return MDCModelOutput(loss=loss, logits=logits)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)

import json
import os

import hydra
import kagglehub
import pandas as pd
import torch
from accelerate.logging import get_logger
from llm_filter.mdc_dataset import MDCDataset
from llm_filter.mdc_loader import MDCCollator, show_batch
from llm_filter.mdc_model import MDCModel, get_base_model
from llm_filter.mdc_optimizer import get_optimizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.train_utils import AverageMeter, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run

logger = get_logger(__name__)
torch._dynamo.config.optimize_ddp = False


@hydra.main(version_base=None, config_path="../conf/llm_filter", config_name="conf_baseline")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.local_rank = accelerator.process_index

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    print_line()
    accelerator.print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    with accelerator.main_process_first():
        input_dir = kagglehub.dataset_download(cfg.dataset.input_dataset)

    fold = cfg.fold

    input_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    aug_df = None  # pd.read_parquet(os.path.join(input_dir, "aug.parquet"))

    if cfg.full_fit:
        train_df = input_df.copy()
    else:
        train_df = input_df[input_df["kfold"] != fold].copy()
    valid_df = input_df[input_df["kfold"] == fold].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    print_line()

    # dataset ----
    train_ds = MDCDataset(cfg, train_df, aug_df=aug_df, seed=cfg.seed)
    valid_ds = MDCDataset(cfg, valid_df, aug_df=aug_df, seed=cfg.seed)
    tokenizer = train_ds.tokenizer

    data_collator = MDCCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()
    for idx, b in enumerate(train_dl):
        accelerator.print(f"TRAINING BATCH {idx}:")
        show_batch(b, tokenizer, task="training", print_fn=accelerator.print)
        if idx > 4:
            break

    # --- model -------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    base_model = get_base_model(cfg)
    model = MDCModel(cfg, base_model, tokenizer)

    if cfg.model.use_gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    accelerator.wait_for_everyone()

    if cfg.model.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model)

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # ------- training setup ---------------------------------------------------------------#
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    accelerator.wait_for_everyone()
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(f"STEP: {current_iteration + 1:5}/{num_training_steps:5}. LR: {get_lr(optimizer):.4f}. Loss: {loss_meter.avg:.4f}. ")

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)
                    accelerator.log({"total_grad_l2_norm": round(grad_norm.item(), 5)}, step=current_iteration)

    # --- end training
    accelerator.wait_for_everyone()
    if cfg.save_model:
        model.eval()

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(cfg.outputs.model_dir)
            tokenizer.save_pretrained(cfg.outputs.model_dir)

            # save the merged model
            if cfg.model.use_lora:
                accelerator.print("saving the merged model...")
                merged_model = unwrapped_model.model.merge_and_unload(safe_merge=True)

                # Save the merged model to a separate directory
                merged_model_dir = os.path.join(cfg.outputs.model_dir, "merged")
                os.makedirs(merged_model_dir, exist_ok=True)

                merged_model.save_pretrained(merged_model_dir)
                tokenizer.save_pretrained(merged_model_dir)

                accelerator.print(f"Merged model saved to: {merged_model_dir}")

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()

# -*- coding: utf-8 -*-
"""ssCDL demo — unKR library wrapper (semi-supervised curriculum distillation, 2025)."""

import argparse
import csv
import os
import signal
import time
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from unKR.utils import *
from unKR.data.Sampler import *


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + timing CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(trainer, model_name, dataset_name, output_dir="results/baselines"):
    os.makedirs(output_dir, exist_ok=True)
    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"metrics_{model_name}_{dataset_name}_{now}.csv")

    # ssCDLLitModel.test_epoch_end uses "Eval" prefix
    m = {k: v.item() if hasattr(v, 'item') else float(v)
         for k, v in trainer.callback_metrics.items()}
    print("Available metric keys:", list(m.keys()))

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relation", "mrr", "wmrr", "wmr", "mr",
            "hits@1", "hits@3", "hits@5", "hits@10",
            "mae", "mse",
        ])
        writer.writerow([
            "overall",
            m.get("Eval_mrr"),
            m.get("Eval_wmrr"),
            m.get("Eval_wmr"),
            m.get("Eval_mr"),
            m.get("Eval_hits@1"),
            m.get("Eval_hits@3"),
            m.get("Eval_hits@5"),
            m.get("Eval_hits@10"),
            m.get("Eval_MAE"),
            m.get("Eval_MSE"),
        ])

    print(f"Metrics saved to: {filepath}")


def save_timing_csv(model_name, dataset_name, total_time_s, status="completed", output_dir="results/baselines"):
    os.makedirs(output_dir, exist_ok=True)
    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"timing_{model_name}_{dataset_name}_{now}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dataset",
            "total_train_time_s", "total_train_time_h", "status",
        ])
        writer.writeheader()
        writer.writerow({
            "model":              model_name,
            "dataset":            dataset_name,
            "total_train_time_s": round(total_time_s, 2),
            "total_train_time_h": round(total_time_s / 3600, 4),
            "status":             status,
        })

    print(f"Timing saved to: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path, dataset_name, seed=None):
    print(f"Running ssCDL — config: {config_path}  dataset: {dataset_name}")

    args = setup_parser()
    args = load_config(args, config_path)

    if dataset_name:
        args.dataset_name = dataset_name
    if seed is not None:
        args.seed = seed

    seed_everything(args.seed)
    print(f"Dataset: {args.dataset_name}")

    model_name = args.model_name

    # ── Data ──────────────────────────────────────────────────────────────────
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler       = train_sampler_class(args)
    test_sampler_class  = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler        = test_sampler_class(train_sampler)

    data_class = import_class(f"unKR.data.{args.data_class}")
    kgdata     = data_class(args, train_sampler, test_sampler)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_class    = import_class(f"unKR.model.{model_name}")
    model          = model_class(args)
    litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
    lit_model      = litmodel_class(model, args)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([model_name, args.dataset_name, str(args.lr)])
        logger   = pl.loggers.WandbLogger(name=log_name, project="unKR")
        logger.log_hyperparams(vars(args))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # ssCDL has curriculum phases: warmup (<30 epochs) + meta-training.
    # Monitor MAE; patience > 30 to survive the warmup phase.
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval_MAE", mode="min",
        patience=args.early_stop_patience,
        check_on_train_epoch_end=False)

    dirpath = "/".join(["output", args.eval_task, args.dataset_name, model_name, f"s{args.seed}", ""])
    last_ckpt = os.path.join(dirpath, "last.ckpt")
    resume_ckpt = None
    if os.path.isfile(last_ckpt):
        try:
            ckpt_state = torch.load(last_ckpt, map_location="cpu").get("state_dict", {})
            ent_w = ckpt_state.get("model.ent_emb.weight")
            if ent_w is not None and ent_w.shape[1] != args.emb_dim:
                print(f"WARNING: checkpoint emb_dim={ent_w.shape[1]} != config emb_dim={args.emb_dim}, ignoring checkpoint.")
            else:
                resume_ckpt = last_ckpt
                print(f"Resuming from checkpoint: {resume_ckpt}")
        except Exception as e:
            print(f"WARNING: could not read checkpoint ({e}), starting from scratch.")

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_MAE", mode="min",
        filename="{epoch}-{Eval_MAE:.5f}",
        dirpath=dirpath,
        save_weights_only=True, save_top_k=1)

    # Keep last checkpoint: ssCDL's pseudo-label quality improves progressively
    model_checkpoint_last = pl.callbacks.ModelCheckpoint(
        filename="last-{epoch}",
        dirpath=dirpath,
        save_weights_only=True, save_top_k=1, save_last=True)

    callbacks = [early_callback, model_checkpoint, model_checkpoint_last]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
    )
    if args.gpu != "cpu":
        trainer_kwargs["gpus"] = "0,"

    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)

    if args.save_config:
        save_config(args)

    # ── Train + Test ──────────────────────────────────────────────────────────
    train_start = time.time()

    def _on_signal(signum, frame):
        raise SystemExit(f"Killed by signal {signum}")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    _status = "interrupted"
    try:
        if not args.test_only:
            trainer.fit(lit_model, datamodule=kgdata, ckpt_path=resume_ckpt)
            # prefer last checkpoint: curriculum distillation improves over time
            path = model_checkpoint_last.last_model_path or model_checkpoint.best_model_path
        else:
            path = args.checkpoint_dir

        lit_model.load_state_dict(torch.load(path)["state_dict"])
        lit_model.eval()
        trainer.test(lit_model, datamodule=kgdata)

        # ── Save outputs ──────────────────────────────────────────────────────
        save_metrics_csv(trainer, f"{model_name}_s{args.seed}", args.dataset_name)
        _status = "completed"
    finally:
        save_timing_csv(f"{model_name}_s{args.seed}", args.dataset_name,
                        time.time() - train_start, status=_status)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ssCDL unKR runner")
    parser.add_argument("--config", type=str,
                        default="baselinesu/config/ppi5k/ssCDL_ppi5k.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (for multiple runs)")
    cli_args = parser.parse_args()

    main(config_path=cli_args.config, dataset_name=cli_args.dataset, seed=cli_args.seed)

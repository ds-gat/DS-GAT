# -*- coding: utf-8 -*-
"""UPGAT demo — unKR library wrapper."""

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
# Training curves callback
# ─────────────────────────────────────────────────────────────────────────────

class TrainingCurveCallback(pl.Callback):
    def __init__(self, label=""):
        super().__init__()
        self.label   = label
        self.records = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        def _v(key):
            val = m.get(key, '')
            return val.item() if hasattr(val, 'item') else val

        self.records.append({
            'epoch':      trainer.current_epoch,
            'train_loss': _v('Train|loss'),
            'val_mrr':    _v('Eval_mrr'),
            'val_wmrr':   _v('Eval_wmrr'),
            'val_wmr':    _v('Eval_wmr'),
            'val_mae':    _v('Eval_MAE'),
        })

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['epoch', 'train_loss', 'val_mrr',
                               'val_wmrr', 'val_wmr', 'val_mae'])
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Training curves ({self.label}) saved to: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + timing CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(trainer, model_name, dataset_name, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"metrics_{model_name}_{dataset_name}_{now}.csv")

    m = {k: v.item() if hasattr(v, 'item') else float(v)
         for k, v in trainer.callback_metrics.items()}

    print("Available metric keys:", list(m.keys()))

    def _get(test_key, eval_key):
        return m.get(test_key) if m.get(test_key) is not None else m.get(eval_key)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relation", "mrr", "wmrr", "wmr", "mr",
            "hits@1", "hits@3", "hits@5", "hits@10",
            "mae", "mse",
        ])
        writer.writerow([
            "overall",
            _get("Test_mrr",    "Eval_mrr"),
            _get("Test_wmrr",   "Eval_wmrr"),
            _get("Test_wmr",    "Eval_wmr"),
            _get("Test_mr",     "Eval_mr"),
            _get("Test_hits@1", "Eval_hits@1"),
            _get("Test_hits@3", "Eval_hits@3"),
            _get("Test_hits@5", "Eval_hits@5"),
            _get("Test_hits@10","Eval_hits@10"),
            _get("Test_MAE",    "Eval_MAE"),
            _get("Test_MSE",    "Eval_MSE"),
        ])

    print(f"Metrics saved to: {filepath}")


def save_timing_csv(model_name, dataset_name, total_time_s, status="completed", output_dir="output"):
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

    print(f"Timing saved to:  {filepath}")


def _build_trainer(args, callbacks, logger):
    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
    )
    if args.gpu != "cpu":
        trainer_kwargs["gpus"] = "0,"
    return pl.Trainer.from_argparse_args(args, **trainer_kwargs)


def _build_data_and_model(args):
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler       = train_sampler_class(args)
    test_sampler_class  = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler        = test_sampler_class(train_sampler)

    data_class = import_class(f"unKR.data.{args.data_class}")
    kgdata     = data_class(args, train_sampler, test_sampler)

    model_class    = import_class(f"unKR.model.{args.model_name}")
    model          = model_class(args)
    litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
    lit_model      = litmodel_class(model, args)

    return train_sampler, kgdata, model, lit_model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path, dataset_name, seed=None):
    print(f"Running UPGAT — config: {config_path}  dataset: {dataset_name}")

    args = setup_parser()
    args = load_config(args, config_path)

    if dataset_name:
        args.dataset_name = dataset_name
    if seed is not None:
        args.seed = seed

    seed_everything(args.seed)
    print(f"Dataset: {args.dataset_name}")

    model_name  = args.model_name
    train_start = time.time()

    def _on_signal(signum, frame):
        raise SystemExit(f"Killed by signal {signum}")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    def make_logger():
        logger = pl.loggers.TensorBoardLogger("training/logs")
        if args.use_wandb:
            log_name = "_".join([model_name, args.dataset_name, str(args.lr)])
            logger   = pl.loggers.WandbLogger(name=log_name, project="unKR")
            logger.log_hyperparams(vars(args))
        return logger

    # ─────────────────────────────────────────────────────────────────────────
    # Teacher model
    # ─────────────────────────────────────────────────────────────────────────
    pseudo_path = "/".join([args.data_path, "pseudo.tsv"])
    if not args.test_only and args.teacher_model and os.path.isfile(pseudo_path):
        print(f"pseudo.tsv already exists at {pseudo_path}, skipping teacher training.")
    elif not args.test_only and args.teacher_model:
        print("--------------------------------")
        print("Teacher model has been started.")

        train_sampler, kgdata, model, lit_model = _build_data_and_model(args)
        teacher_curves = TrainingCurveCallback(label="teacher")

        dirpath_t = "/".join(["output", args.eval_task,
                               args.dataset_name, model_name, f"s{args.seed}", "teacher_model"])

        # If a best (completed) teacher checkpoint exists, load it directly and skip training
        existing_best = sorted([f for f in os.listdir(dirpath_t)
                                 if f.endswith(".ckpt") and f != "last.ckpt"]) \
                        if os.path.isdir(dirpath_t) else []
        if existing_best:
            path = os.path.join(dirpath_t, existing_best[0])
            print(f"Teacher checkpoint found, skipping training. Loading: {path}")
            lit_model.load_state_dict(torch.load(path)["state_dict"])
        else:
            early_cb = pl.callbacks.EarlyStopping(
                monitor="Eval_MAE", mode="min",
                patience=args.early_stop_patience,
                check_on_train_epoch_end=False)

            ckpt_cb = pl.callbacks.ModelCheckpoint(
                monitor="Eval_MAE", mode="min",
                filename="{epoch}-{Eval_MAE:.5f}",
                dirpath=dirpath_t,
                save_weights_only=True, save_top_k=1, save_last=True)

            trainer_t = _build_trainer(
                args, [early_cb, ckpt_cb, teacher_curves], make_logger())

            if args.save_config:
                save_config(args)

            trainer_t.fit(lit_model, datamodule=kgdata)
            path = ckpt_cb.best_model_path
            lit_model.load_state_dict(torch.load(path)["state_dict"])

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            teacher_curves.save(
                f"output/training_curves_{model_name}_{args.dataset_name}_teacher_{now}.csv")

        print("Teacher model ready.")
        print("--------------------------------")

        model = model.to(args.gpu)
        output_file_path = "/".join([args.data_path, "pseudo.tsv"])
        model.pseudo_tail_predict(train_sampler.train_triples, output_file_path)
        print(f"Pseudo data written to {output_file_path}.")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # ─────────────────────────────────────────────────────────────────────────
    # Student model
    # ─────────────────────────────────────────────────────────────────────────
    print("--------------------------------")
    print("Student model has been started.")

    args.teacher_model = False
    _, kgdata, _, lit_model = _build_data_and_model(args)
    student_curves = TrainingCurveCallback(label="student")

    early_cb = pl.callbacks.EarlyStopping(
        monitor="Eval_MAE", mode="min",
        patience=args.early_stop_patience,
        check_on_train_epoch_end=False)

    dirpath_s = "/".join(["output", args.eval_task,
                           args.dataset_name, model_name, f"s{args.seed}", "student_model"])
    ckpt_cb   = pl.callbacks.ModelCheckpoint(
        monitor="Eval_wmr", mode="min",
        filename="{epoch}-{Eval_wmr:.5f}",
        dirpath=dirpath_s,
        save_weights_only=True, save_top_k=1, save_last=True)

    trainer_s = _build_trainer(
        args, [early_cb, ckpt_cb, student_curves], make_logger())

    if args.save_config:
        save_config(args)

    last_ckpt_s = os.path.join(dirpath_s, "last.ckpt")
    resume_ckpt_s = None
    if os.path.isfile(last_ckpt_s):
        try:
            ckpt_state = torch.load(last_ckpt_s, map_location="cpu").get("state_dict", {})
            ent_w = ckpt_state.get("model.ent_emb")
            if ent_w is not None and ent_w.shape[1] != args.emb_dim:
                print(f"WARNING: student checkpoint emb_dim={ent_w.shape[1]} != config emb_dim={args.emb_dim}, ignoring checkpoint.")
            else:
                resume_ckpt_s = last_ckpt_s
                print(f"Resuming student from checkpoint: {resume_ckpt_s}")
        except Exception as e:
            print(f"WARNING: could not read student checkpoint ({e}), starting from scratch.")

    _status = "interrupted"
    try:
        if not args.test_only:
            trainer_s.fit(lit_model, datamodule=kgdata, ckpt_path=resume_ckpt_s)
            path = ckpt_cb.best_model_path
        else:
            # Find best checkpoint (lowest Eval_wmr) in the student model dir
            ckpt_files = [f for f in os.listdir(dirpath_s) if f.endswith(".ckpt") and f != "last.ckpt"]
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {dirpath_s}")
            path = os.path.join(dirpath_s, sorted(ckpt_files)[0])
            print(f"Eval-only: loading checkpoint {path}")

        lit_model.load_state_dict(torch.load(path)["state_dict"])
        lit_model.eval()
        trainer_s.test(lit_model, datamodule=kgdata)

        # ── Save outputs ──────────────────────────────────────────────────────
        now      = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_tag = f"{model_name}_s{args.seed}_{args.dataset_name}"

        save_metrics_csv(trainer_s, f"{model_name}_s{args.seed}", args.dataset_name)
        student_curves.save(f"output/training_curves_{file_tag}_student_{now}.csv")
        _status = "completed"
    finally:
        save_timing_csv(f"{model_name}_s{args.seed}", args.dataset_name,
                        time.time() - train_start, status=_status)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPGAT unKR runner")
    parser.add_argument("--config", type=str,
                        default="config/ppi5k/UPGAT_ppi5k.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (for multiple runs)")
    cli_args = parser.parse_args()

    main(config_path=cli_args.config, dataset_name=cli_args.dataset, seed=cli_args.seed)
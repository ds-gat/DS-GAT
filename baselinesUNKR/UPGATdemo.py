# -*- coding: utf-8 -*-
"""UPGAT demo — unKR library wrapper."""

import argparse
import csv
import os
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

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relation", "mrr", "wmrr", "wmr", "mr",
            "hits@1", "hits@3", "hits@5", "hits@10",
            "mae", "mse",
        ])
        writer.writerow([
            "overall",
            m.get("Test_mrr"),
            m.get("Test_wmrr"),
            m.get("Test_wmr"),
            m.get("Test_mr"),
            m.get("Test_hits@1"),
            m.get("Test_hits@3"),
            m.get("Test_hits@5"),
            m.get("Test_hits@10"),
            m.get("Test_MAE"),
            m.get("Test_MSE"),
        ])

    print(f"Metrics saved to: {filepath}")


def save_timing_csv(model_name, dataset_name, total_time_s, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"timing_{model_name}_{dataset_name}_{now}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dataset",
            "total_train_time_s", "total_train_time_h",
        ])
        writer.writeheader()
        writer.writerow({
            "model":              model_name,
            "dataset":            dataset_name,
            "total_train_time_s": round(total_time_s, 2),
            "total_train_time_h": round(total_time_s / 3600, 4),
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

def main(config_path, dataset_name):
    print(f"Running UPGAT — config: {config_path}  dataset: {dataset_name}")

    args = setup_parser()
    args = load_config(args, config_path)

    if dataset_name:
        args.dataset_name = dataset_name

    seed_everything(args.seed)
    print(f"Dataset: {args.dataset_name}")

    model_name  = args.model_name
    train_start = time.time()

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
    if not args.test_only and args.teacher_model:
        print("--------------------------------")
        print("Teacher model has been started.")

        train_sampler, kgdata, model, lit_model = _build_data_and_model(args)
        teacher_curves = TrainingCurveCallback(label="teacher")

        early_cb = pl.callbacks.EarlyStopping(
            monitor="Eval_MAE", mode="min",
            patience=args.early_stop_patience,
            check_on_train_epoch_end=False)

        dirpath_t = "/".join(["output", args.eval_task,
                               args.dataset_name, model_name, "teacher_model"])
        ckpt_cb   = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MAE", mode="min",
            filename="{epoch}-{Eval_MAE:.5f}",
            dirpath=dirpath_t,
            save_weights_only=True, save_top_k=1)

        trainer_t = _build_trainer(
            args, [early_cb, ckpt_cb, teacher_curves], make_logger())

        if args.save_config:
            save_config(args)

        trainer_t.fit(lit_model, datamodule=kgdata)
        path = ckpt_cb.best_model_path
        lit_model.load_state_dict(torch.load(path)["state_dict"])
        lit_model.eval()
        trainer_t.test(lit_model, datamodule=kgdata)

        # Save teacher curves
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        teacher_curves.save(
            f"output/training_curves_{model_name}_{args.dataset_name}_teacher_{now}.csv")

        print("Teacher model has been trained.")
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
                           args.dataset_name, model_name, "student_model"])
    ckpt_cb   = pl.callbacks.ModelCheckpoint(
        monitor="Eval_wmr", mode="min",
        filename="{epoch}-{Eval_wmr:.5f}",
        dirpath=dirpath_s,
        save_weights_only=True, save_top_k=1)

    trainer_s = _build_trainer(
        args, [early_cb, ckpt_cb, student_curves], make_logger())

    if args.save_config:
        save_config(args)

    if not args.test_only:
        trainer_s.fit(lit_model, datamodule=kgdata)
        path = ckpt_cb.best_model_path
    else:
        path = "./output/confidence_prediction/ppi5k/UPGAT/student_model/epoch=85-Eval_wmr=2.38164.ckpt"

    total_train_time = time.time() - train_start

    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer_s.test(lit_model, datamodule=kgdata)

    # ── Save outputs ──────────────────────────────────────────────────────────
    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_tag = f"{model_name}_{args.dataset_name}"

    save_metrics_csv(trainer_s, model_name, args.dataset_name)
    save_timing_csv(model_name, args.dataset_name, total_train_time)
    student_curves.save(f"output/training_curves_{file_tag}_student_{now}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPGAT unKR runner")
    parser.add_argument("--config", type=str,
                        default="config/ppi5k/UPGAT_ppi5k.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    cli_args = parser.parse_args()

    main(config_path=cli_args.config, dataset_name=cli_args.dataset)
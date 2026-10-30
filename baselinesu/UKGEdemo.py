# -*- coding: utf-8 -*-
"""UKGE demo — unKR library wrapper (Chen et al., AAAI 2019; logistic UKGE).

Same conventions as the other baselines (FocusEdemo, UPGATdemo, ssCDLdemo):
  * --config / --dataset / --seed from the CLI
  * output/metrics_UKGE_s{seed}_{dataset}_{timestamp}.csv
        relation, mrr, wmrr, wmr, mr, hits@1,3,5,10, mae, mse   (test, tail, filtered)
  * output/timing_UKGE_s{seed}_{dataset}_{timestamp}.csv   (cumulative over resumes)
  * output/training_curves_UKGE_s{seed}_{dataset}_{timestamp}.csv
  * checkpoints in output/confidence_prediction/{dataset}/UKGE/s{seed}/
Model selection: best validation MAE (unKR's UKGE default). A job killed at the
2-day limit resumes from last.ckpt when resubmitted with the same seed.
Resubmitting a seed is always safe: if its metrics file already exists the run is
skipped, and if training had finished (early stopping or max_epochs) but the test
was not saved, only the test is run on the best checkpoint.
"""

import argparse
import glob
import csv
import inspect
import os
import signal
import time
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from unKR.utils import *
from unKR.data.Sampler import *


class TrainingCurveCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        def _v(key):
            val = m.get(key, '')
            return val.item() if hasattr(val, 'item') else val
        self.records.append({'epoch': trainer.current_epoch,
                             'train_loss': _v('Train|loss'),
                             'val_mrr': _v('Eval_mrr'), 'val_wmrr': _v('Eval_wmrr'),
                             'val_mae': _v('Eval_MAE'), 'val_mse': _v('Eval_MSE')})

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_mrr', 'val_wmrr',
                                              'val_mae', 'val_mse'])
            w.writeheader()
            w.writerows(self.records)
        print(f"Training curves saved to: {filepath}")


def save_metrics_csv(trainer, model_name, dataset_name, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"metrics_{model_name}_{dataset_name}_{now}.csv")
    m = {k: v.item() if hasattr(v, 'item') else float(v)
         for k, v in trainer.callback_metrics.items()}
    print("Available metric keys:", list(m.keys()))

    def _get(key):  # UKGELitModel logs test metrics with the "Test" prefix
        return m.get(f"Test_{key}") if m.get(f"Test_{key}") is not None else m.get(f"Eval_{key}")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relation", "mrr", "wmrr", "wmr", "mr",
                         "hits@1", "hits@3", "hits@5", "hits@10", "mae", "mse"])
        writer.writerow(["overall", _get("mrr"), _get("wmrr"), _get("wmr"), _get("mr"),
                         _get("hits@1"), _get("hits@3"), _get("hits@5"), _get("hits@10"),
                         _get("MAE"), _get("MSE")])
    print(f"Metrics saved to: {filepath}")


def save_timing_csv(model_name, dataset_name, total_time_s, status="completed",
                    output_dir="output", segment_time_s=None, n_segments=1):
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"timing_{model_name}_{dataset_name}_{now}.csv")
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "total_train_time_s",
                                          "total_train_time_h", "status",
                                          "segment_time_s", "n_segments"])
        w.writeheader()
        w.writerow({"model": model_name, "dataset": dataset_name,
                    "total_train_time_s": round(total_time_s, 2),
                    "total_train_time_h": round(total_time_s / 3600, 4),
                    "status": status,
                    "segment_time_s": round(total_time_s if segment_time_s is None else segment_time_s, 2),
                    "n_segments": n_segments})
    print(f"Timing saved to:  {filepath}")


def main(config_path, dataset_name, seed=None):
    print(f"Running UKGE — config: {config_path}  dataset: {dataset_name}")
    args = setup_parser()
    args = load_config(args, config_path)
    if dataset_name:
        args.dataset_name = dataset_name
    if seed is not None:
        args.seed = seed
    seed_everything(args.seed)
    model_name = args.model_name

    done = glob.glob(f"output/metrics_{model_name}_s{args.seed}_{args.dataset_name}_*.csv")
    if done:
        print(f"Already finished, skipping: {done[0]}")
        return

    train_sampler = import_class(f"unKR.data.{args.train_sampler_class}")(args)
    test_sampler = import_class(f"unKR.data.{args.test_sampler_class}")(train_sampler)
    kgdata = import_class(f"unKR.data.{args.data_class}")(args, train_sampler, test_sampler)
    model = import_class(f"unKR.model.{model_name}")(args)
    lit_model = import_class(f"unKR.lit_model.{args.litmodel_name}")(model, args)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    dirpath = "/".join(["output", args.eval_task, args.dataset_name, model_name, f"s{args.seed}"])
    early_cb = pl.callbacks.EarlyStopping(monitor="Eval_MAE", mode="min",
                                          patience=args.early_stop_patience,
                                          check_on_train_epoch_end=False)
    # full checkpoints (optimizer + epoch) so a job killed at the limit can resume
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="Eval_MAE", mode="min",
                                           filename="{epoch}-{Eval_MAE:.5f}", dirpath=dirpath,
                                           save_weights_only=False, save_top_k=1, save_last=True)
    curves = TrainingCurveCallback()
    trainer_kwargs = dict(callbacks=[early_cb, ckpt_cb, curves], logger=logger,
                          default_root_dir="training/logs",
                          check_val_every_n_epoch=args.check_val_every_n_epoch,
                          max_epochs=args.max_epochs)
    if args.gpu != "cpu":
        trainer_kwargs["gpus"] = "0,"
    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)

    last_ckpt = os.path.join(dirpath, "last.ckpt")
    resume_ckpt = None
    eval_only_ckpt = None
    if os.path.isfile(last_ckpt):
        try:
            full = torch.load(last_ckpt, map_location="cpu")
            if "optimizer_states" in full:
                resume_ckpt = last_ckpt
                print(f"Resuming from checkpoint: {resume_ckpt}")
                # training already over (early stopping or max_epochs): only test
                es = next((v for k, v in full.get("callbacks", {}).items()
                           if "EarlyStopping" in str(k)), {})
                stopped = es.get("wait_count", 0) >= es.get("patience", args.early_stop_patience)
                if stopped or full.get("epoch", 0) + 1 >= args.max_epochs:
                    best = sorted(glob.glob(os.path.join(dirpath, "epoch=*.ckpt")))
                    if best:
                        eval_only_ckpt = best[-1]
                        print(f"Training already finished, test only: {eval_only_ckpt}")
            else:
                print("WARNING: last.ckpt is weights-only, starting from scratch.")
        except Exception as e:
            print(f"WARNING: could not read checkpoint ({e}), starting from scratch.")

    train_start = time.time()

    def _on_signal(signum, frame):
        raise SystemExit(f"Killed by signal {signum}")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    status = "interrupted"
    try:
        if eval_only_ckpt:
            lit_model.load_state_dict(torch.load(eval_only_ckpt, map_location="cpu")["state_dict"])
            lit_model.eval()
            trainer.test(lit_model, datamodule=kgdata)
            save_metrics_csv(trainer, f"{model_name}_s{args.seed}", args.dataset_name)
            status = "completed"
        elif "ckpt_path" in inspect.signature(trainer.fit).parameters:
            trainer.fit(lit_model, datamodule=kgdata, ckpt_path=resume_ckpt)
        else:  # Lightning < 1.5
            if resume_ckpt:
                args.resume_from_checkpoint = resume_ckpt
                trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)
            trainer.fit(lit_model, datamodule=kgdata)
        if not eval_only_ckpt:
            lit_model.load_state_dict(torch.load(ckpt_cb.best_model_path)["state_dict"])
            lit_model.eval()
            trainer.test(lit_model, datamodule=kgdata)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_metrics_csv(trainer, f"{model_name}_s{args.seed}", args.dataset_name)
            curves.save(f"output/training_curves_{model_name}_s{args.seed}_{args.dataset_name}_{now}.csv")
            status = "completed"
    finally:
        if status != "completed" and curves.records:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            curves.save(f"output/training_curves_{model_name}_s{args.seed}_{args.dataset_name}_partial_{now}.csv")
        # a test-only run adds no training time and no segment
        segment_s = 0.0 if eval_only_ckpt else time.time() - train_start
        elapsed_file = os.path.join(dirpath, "elapsed_s.txt")
        prev_s, n_seg = 0.0, 0
        if os.path.isfile(elapsed_file):
            with open(elapsed_file) as f:
                prev_s, n_seg = f.read().split()
            prev_s, n_seg = float(prev_s), int(n_seg)
        if eval_only_ckpt:
            n_seg -= 1
        os.makedirs(dirpath, exist_ok=True)
        with open(elapsed_file, "w") as f:
            f.write(f"{prev_s + segment_s} {n_seg + 1}")
        save_timing_csv(f"{model_name}_s{args.seed}", args.dataset_name, prev_s + segment_s,
                        status=status, segment_time_s=segment_s, n_segments=n_seg + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UKGE unKR runner")
    parser.add_argument("--config", type=str, default="baselinesu/config/cn15k/UKGE_cn15k.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    cli_args = parser.parse_args()
    main(config_path=cli_args.config, dataset_name=cli_args.dataset, seed=cli_args.seed)

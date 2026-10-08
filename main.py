"""Run the de-identified CFAF-DDCT synthetic-data example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfaf_ddct.config import Config
from cfaf_ddct.data import seed_everything
from cfaf_ddct.train import run
from synthetic_data import load_experiment_data


def parse_args():
    parser = argparse.ArgumentParser(description="CFAF-DDCT synthetic runnable reference")
    parser.add_argument("--loss", choices=["none", "icl", "mmd", "lmmd", "coral", "adver"], default="icl")
    parser.add_argument("--epochs", type=int, choices=[200], default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="source/target sample ratio used for feature adaptation")
    parser.add_argument("--no-normalize-transfer-features", action="store_true",
                        help="keep raw feature scale when computing adaptation loss")
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for automatic selection")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs"),
                        help="folder containing synthetic_data.npz")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="folder for result.json; defaults to --data-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(loss=args.loss, fine_tune_epochs=args.epochs, seed=args.seed,
                    source_target_ratio=args.gamma,
                    normalize_transfer_features=not args.no_normalize_transfer_features)
    seed_everything(args.seed)
    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] result folder={output_dir.resolve()}")
    data_path = args.data_dir / "synthetic_data.npz"
    print(f"[data] loading synthetic data from {data_path.resolve()}")
    source_subjects, target_x, target_y = load_experiment_data(data_path)
    source_x = np.concatenate([x for x, _ in source_subjects], axis=0)
    source_y = np.concatenate([y for _, y in source_subjects], axis=0)
    print(f"[data] source_x={tuple(source_x.shape)} source_y={tuple(source_y.shape)} "
          f"target_x={tuple(target_x.shape)} target_y={tuple(target_y.shape)}")
    print(f"[labels] source_counts={np.bincount(source_y, minlength=3).tolist()} "
          f"target_counts={np.bincount(target_y, minlength=3).tolist()}")
    result = run(source_subjects, target_x, target_y, config,
                 device=args.device, verbose=True)
    result["config"] = {
        "loss": config.loss,
        "fine_tune_epochs": config.fine_tune_epochs,
        "source_pretrain_epochs": config.source_pretrain_epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "retained_channels": config.retained_channels,
        "lambda": config.lambda_value,
        "gamma": config.source_target_ratio,
        "normalize_transfer_features": config.normalize_transfer_features,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[output] result saved to {result_path}")
    print("CFAF-DDCT synthetic run completed")
    print(f"loss={result['loss']} lambda={result['lambda']:.3f} "
          f"validation_acc={result['validation_accuracy']:.3f} "
          f"test_acc={result['test_accuracy']:.3f}")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

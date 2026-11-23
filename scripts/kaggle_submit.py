"""Submit training-run submissions to the Kaggle competition.

Credentials come from ``KAGGLE_API_TOKEN`` in ``<repo>/.env``, loaded with
python-dotenv; the token value is never printed. Each ``--exp-name`` maps to
``storage/artifacts/training/<exp_name>/submission.csv``; the submission
message is built from the run's config (dataset version, feature count) and
its OOF ROC AUC from ``eval_off/metrics.csv``.

Usage::

    python scripts/kaggle_submit.py --exp-name hgb_no_time
    python scripts/kaggle_submit.py --exp-name hgb_no_time --exp-name hgb_no_time_no_channel
    python scripts/kaggle_submit.py --list-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ccf_ml.file_utils import (
    get_repo_root,
    get_training_config_path,
    get_training_eval_dir,
    get_training_oof_path,
    get_training_submission_path,
)

COMPETITION = "playground-series-s6e11"
METRICS_FILENAME = "metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        action="append",
        help="Training run to submit (repeatable).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the competition's recent submissions.",
    )
    args = parser.parse_args()

    env_path = get_repo_root() / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"missing .env file: {env_path}")
    load_dotenv(env_path, override=True)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise ValueError(
            f"KAGGLE_API_TOKEN is not set in {env_path}; "
            "add it there or export it before submitting."
        )
    print(f"credentials: KAGGLE_API_TOKEN loaded from .env (length {len(token)})")

    if args.list_only:
        _list_submissions()
        return

    exp_names = list(dict.fromkeys(args.exp_name or []))
    if not exp_names:
        raise ValueError("nothing to do: pass --exp-name or --list-only")

    for exp_name in exp_names:
        _submit_exp(exp_name)
    _list_submissions()
    print("done")


def _submit_exp(exp_name: str) -> None:
    submission_path = get_training_submission_path(exp_name)
    if not submission_path.is_file():
        raise FileNotFoundError(f"no submission.csv for {exp_name!r}: {submission_path}")
    message = _build_message(exp_name)
    print(f"\n[{exp_name}] submitting {_repo_relative(submission_path)}")
    print(f"  message: {message}")
    _run_kaggle(
        [
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(submission_path),
            "-m",
            message,
        ]
    )


def _build_message(exp_name: str) -> str:
    config = json.loads(get_training_config_path(exp_name).read_text(encoding="utf-8"))
    version = config.get("dataset_version")
    bag = config.get("bag")
    if bag:
        parts = [exp_name, f"bag of {len(bag)}"]
    else:
        n_features = len(config.get("columns", {}).get("features", []))
        parts = [exp_name, f"{n_features} feats"]
    if version:
        parts.append(f"version={version['name']}")
    oof_auc = _read_oof_auc(exp_name)
    if oof_auc is not None:
        parts.append(f"OOF {oof_auc:.5f}")
    return " | ".join(parts)


def _read_oof_auc(exp_name: str) -> float | None:
    metrics_path = get_training_eval_dir(exp_name) / METRICS_FILENAME
    if not metrics_path.is_file():
        return None
    metrics = pd.read_csv(metrics_path)
    overall = metrics[metrics["fold_id"] == "oof"]
    if overall.empty:
        return None
    return float(overall["roc_auc"].iloc[0])


def _list_submissions() -> None:
    print("\nrecent submissions:")
    _run_kaggle(["competitions", "submissions", "-c", COMPETITION])


def _run_kaggle(argv: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "kaggle", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        raise RuntimeError(f"kaggle {' '.join(argv[:2])} failed: {detail}")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

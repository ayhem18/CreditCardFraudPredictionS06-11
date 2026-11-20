"""Train no-alarm region models over the variant ladder.

One experiment per (variant, family): the frozen profile-stratified folds,
the variant's model matrix, five fold models, an OOF table, and a config -
the same artifact layout as the global trainers, so ``eval_oof.py`` and the
comparison tooling work unchanged. No test submission: this line is a
region experiment; integration (alarm rows + score alignment) comes later.

Usage::

    python experiments/baselines/train_region_variants.py --family hgb
    python experiments/baselines/train_region_variants.py --family xgb
    python experiments/baselines/train_region_variants.py --family hgb \
        --variant v4_profile_id_quantiles
    python experiments/baselines/train_region_variants.py --family xgb \
        --cv alarm_profile --exp-name-prefix rgnU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from xgboost import XGBClassifier

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.file_utils import (
    claim_training_exp_dir,
    get_repo_root,
    get_train_csv_path,
    get_training_config_path,
    get_training_fold_model_path,
    get_training_oof_path,
)
from ccf_ml.region_modeling import (
    build_no_alarm_frame,
    load_alarm_profile_stratified_cv,
    load_profile_stratified_cv,
    restrict_to_no_alarm,
    variant_by_name,
    variant_model_matrix,
)
from ccf_ml.region_modeling.variants import REGION_MODEL_VARIANTS
from training_io import (
    OOF_FOLD_COL,
    OOF_ID_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    write_config,
    write_oof,
)

FAMILIES = {
    "hgb": {
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "args": {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 42,
            "early_stopping": False,
            "categorical_features": "from_dtype",
        },
    },
    "xgb": {
        "model": "xgboost.XGBClassifier",
        "args": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "n_jobs": -1,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument(
        "--variant",
        action="append",
        choices=[spec.name for spec in REGION_MODEL_VARIANTS],
        help="Train only this variant (repeatable). Default: the whole ladder.",
    )
    parser.add_argument(
        "--cv",
        choices=["profile", "alarm_profile"],
        default="profile",
        help=(
            "Frozen fold scheme: profile (region-only draw) or alarm_profile "
            "(the all-rows scheme shared with the global trainers, restricted "
            "to the region - same fold id per row, directly comparable OOF)."
        ),
    )
    parser.add_argument(
        "--exp-name-prefix",
        default="rgn",
        help="Experiment-name prefix; distinguish reruns under another cv scheme.",
    )
    args = parser.parse_args()

    family = FAMILIES[args.family]
    variants = [
        variant_by_name(name)
        for name in (args.variant or [spec.name for spec in REGION_MODEL_VARIANTS])
    ]
    train_frame = pd.read_csv(get_train_csv_path())
    region = build_no_alarm_frame(train_frame)
    if args.cv == "profile":
        cv_scheme = "profile_stratified"
        cv = load_profile_stratified_cv()
    else:
        cv_scheme = "alarm_profile_stratified"
        cv = restrict_to_no_alarm(load_alarm_profile_stratified_cv(), region)
    print(f"cv: {cv.n_splits}-fold {cv_scheme}, n_rows={cv.n_rows}")
    if len(region) != cv.n_rows:
        raise ValueError(
            f"region has {len(region)} rows; frozen cv has {cv.n_rows}"
        )
    print(f"region: {len(region)} rows, {int(region[TARGET_COLUMN].sum())} frauds")

    results: list[dict[str, object]] = []
    for spec in variants:
        exp_name = f"{args.exp_name_prefix}_{spec.name}_{args.family}"
        print(f"\n[{exp_name}] {spec.description}")
        matrix, feature_columns, categorical_columns, numeric_columns = (
            variant_model_matrix(region, spec, cv=cv)
        )
        _train_one(
            exp_name=exp_name,
            family_name=args.family,
            family=family,
            matrix=matrix,
            cv=cv,
            cv_scheme=cv_scheme,
            spec=spec,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        oof = pd.read_csv(get_training_oof_path(exp_name))
        auc = float(roc_auc_score(oof["is_fraud"], oof["y_prob"]))
        results.append({"exp": exp_name, "variant": spec.name, "oof_auc": round(auc, 5)})
        print(f"  oof auc: {auc:.5f}")

    print("\nsummary")
    print(pd.DataFrame(results).to_string(index=False))
    print("reference: global hgb_baseline on its no-alarm rows: 0.92673")


def _train_one(
    *,
    exp_name: str,
    family_name: str,
    family: dict[str, object],
    matrix: pd.DataFrame,
    cv,
    cv_scheme: str,
    spec,
    feature_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> None:
    """Fit the five fold models, write config / oof / models for one exp."""
    exp_dir = claim_training_exp_dir(exp_name)
    config_path = get_training_config_path(exp_name)
    oof_path = get_training_oof_path(exp_name)
    print(f"  exp_dir: {_repo_relative(exp_dir)}")
    if family_name == "xgb":
        features = _xgb_categories(matrix, categorical_columns)
    else:
        features = matrix.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    y = matrix[TARGET_COLUMN].to_numpy(dtype=np.int64)
    ids = matrix[ID_COLUMN].to_numpy(dtype=np.int64)
    id_index = pd.Index(ids)
    write_config(
        path=config_path,
        payload={
            "exp_name": exp_name,
            "model": family["model"],
            "region": "no_alarm",
            "variant": {
                "name": spec.name,
                "drop_columns": list(spec.drop_columns),
                "add_profile_id": spec.add_profile_id,
                "add_profile_quantiles": spec.add_profile_quantiles,
            },
            "args": family["args"],
            "cv": {
                "scheme": cv_scheme,
                "n_splits": cv.n_splits,
                "random_state": cv.random_state,
            },
            "columns": {
                "id": ID_COLUMN,
                "target": TARGET_COLUMN,
                "features": list(feature_columns),
                "categorical": list(categorical_columns),
                "numeric": list(numeric_columns),
            },
        },
    )
    print(f"  wrote config: {_repo_relative(config_path)}")
    if family_name == "hgb":
        template = sklearn.ensemble.HistGradientBoostingClassifier(**family["args"])
    else:
        template = XGBClassifier(**family["args"])
    oof_prob = np.full(len(ids), np.nan, dtype=np.float64)
    oof_fold = np.full(len(ids), -1, dtype=np.int64)
    t_cv = time.perf_counter()
    for fold in tqdm(cv.folds, desc=f"{exp_name} folds", total=cv.n_splits):
        train_pos = _positions(id_index, ids=fold.train_ids, fold_id=fold.fold_id)
        val_pos = _positions(id_index, ids=fold.val_ids, fold_id=fold.fold_id)
        model = clone(template)
        model.fit(features.iloc[train_pos], y[train_pos])
        oof_prob[val_pos] = model.predict_proba(features.iloc[val_pos])[:, 1]
        oof_fold[val_pos] = fold.fold_id
        joblib.dump(
            model,
            get_training_fold_model_path(exp_name, fold_id=fold.fold_id),
        )
    if np.isnan(oof_prob).any() or np.any(oof_fold < 0):
        raise ValueError("OOF assignment did not cover every region row")
    write_oof(
        path=oof_path,
        frame=pd.DataFrame(
            {
                OOF_ID_COL: ids,
                OOF_FOLD_COL: oof_fold,
                OOF_TARGET_COL: y,
                OOF_PROB_COL: oof_prob,
            }
        ),
    )
    print(f"  wrote oof: {_repo_relative(oof_path)} ({time.perf_counter() - t_cv:.1f}s)")


def _xgb_categories(
    matrix: pd.DataFrame, categorical_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Feature frame for XGBoost: category levels cast to strings."""
    features = matrix.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    for column in categorical_columns:
        features[column] = features[column].astype(str).astype("category")
    return features


def _positions(id_index: pd.Index, *, ids, fold_id: int) -> np.ndarray:
    positions = id_index.get_indexer(ids)
    if np.any(positions < 0):
        raise KeyError(f"fold {fold_id} contains ids missing from the region frame")
    return positions


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

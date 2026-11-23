"""Train XGBoost on a frozen stratified CV (all rows).

Default scheme is flag_stratified (alarm x target); ``--cv alarm_profile``
scores on the unified alarm x profile x target scheme shared with the
region trainers, making the OOF tables of both approaches comparable.

Same artifacts and CLI shape as ``train_hist_gb.py`` (OOF table, one joblib
per fold, averaged test submission); only the model family differs.
Categorical columns ride through as pandas categories with XGBoost's native
categorical splits (``enable_categorical=True``).

Does not score the run. Use ``eval_oof.py`` on the same ``--exp-name``.

Usage::

    python experiments/baselines/train_xgb.py --exp-name xgb_baseline
    python experiments/baselines/train_xgb.py --exp-name xgb_no_time \
        --dataset-version no_time
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm
from xgboost import XGBClassifier

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.cross_validation.flag_stratified import load_flag_stratified_cv
from ccf_ml.region_modeling import load_alarm_profile_stratified_cv
from ccf_ml.dataset_versions import DatasetVersion, load_dataset_version
from ccf_ml.dataset_versions.constants import (
    AUTH_METHOD_COLUMN,
    IS_NEW_MERCHANT_COLUMN,
    MERCHANT_CATEGORY_COLUMN,
    MERCHANT_RISK_SCORE_COLUMN,
    PROFILE_CONTEXT_SIZE_COLUMN,
    ANCHOR_COMBINED_FILENAME,
    ANCHOR_FREQ_FILENAME,
    ANCHOR_RATE_FILENAME,
    ANCHOR_TILT_FILENAME,
    ANCHOR_V2_RATES_FILENAME,
    ANCHOR_V3_RATES_FILENAME,
    IC_CELLZ_FILENAME,
    IC_RATES_FILENAME,
    IC_RISKZ_FINE_FILENAME,
    IS_ALARM_COLUMN,
    N_ALARM_FLAGS_COLUMN,
    SLOT_REFS_FILENAME,
    SLOT_Z_STATS_FILENAME,
    RISK_CONTEXT_ADDED_COLUMNS,
    RISK_CONTEXT_GROUPS_FILENAME,
)
from ccf_ml.dataset_versions.transforms import (
    apply_grid_slots_to_test,
    apply_profile_context_to_test,
    apply_anchor_combined_to_test,
    apply_anchor_freq_to_test,
    apply_anchor_rate_to_test,
    apply_anchor_tilt_to_test,
    apply_anchor_v2_rates_to_test,
    apply_anchor_v3_rates_to_test,
    apply_cellz_to_test,
    apply_ic_rates_to_test,
    apply_riskz_fine_to_test,
    attach_is_alarm,
    attach_n_alarm_flags,
    apply_slot_refs_to_test,
    apply_slot_z_to_test,
)
from ccf_ml.file_utils import (
    claim_training_exp_dir,
    get_repo_root,
    get_test_csv_path,
    get_train_csv_path,
    get_training_config_path,
    get_training_fold_model_path,
    get_training_oof_path,
    get_training_submission_path,
)
from features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    category_levels,
    prepare_features,
)
from train_hist_gb import _apply_risk_context_to_test, _column_lists
from training_io import (
    OOF_FOLD_COL,
    OOF_ID_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    write_config,
    write_oof,
)

MODEL_NAME = "xgboost.XGBClassifier"
CV_SCHEMES = {
    "flag": ("flag_stratified", load_flag_stratified_cv),
    "alarm_profile": ("alarm_profile_stratified", load_alarm_profile_stratified_cv),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Identity of this training run. Must not already exist.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument(
        "--random-state", type=int, default=42
    )
    parser.add_argument(
        "--cv",
        choices=sorted(CV_SCHEMES),
        default="flag",
        help=(
            "Frozen fold scheme to score on: flag (alarm x target, all rows) "
            "or alarm_profile (adds the 90-cell profile to the stratum; the "
            "scheme shared with the region trainers)."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help=(
            "Build the model matrix from this dataset version instead of "
            "train.csv (same frozen folds; test drops the same columns)."
        ),
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    exp_dir = claim_training_exp_dir(exp_name)
    config_path = get_training_config_path(exp_name)
    oof_path = get_training_oof_path(exp_name)
    model_args = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "random_state": args.random_state,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "auc",
        "n_jobs": -1,
    }
    print("train_xgb")
    print(f"exp_name: {exp_name}")
    print(f"exp_dir: {_repo_relative(exp_dir)}")
    print(f"model: {MODEL_NAME}")
    print(f"args: {model_args}")

    dataset_version: DatasetVersion | None = None
    if args.dataset_version is not None:
        dataset_version = load_dataset_version(args.dataset_version)
    feature_columns, categorical_columns, numeric_columns = _column_lists(
        dataset_version=dataset_version,
    )

    if dataset_version is None:
        train_path = get_train_csv_path()
        print(f"train_csv: {_repo_relative(train_path)}")
        frame = pd.read_csv(train_path)
    else:
        manifest = dataset_version.manifest
        print(f"dataset_version: {manifest.name}")
        print(f"  drop_columns: {list(manifest.drop_columns)}")
        print(f"  data_sha256: {manifest.data_sha256}")
        frame = dataset_version.frame
    cv_scheme, load_cv = CV_SCHEMES[args.cv]
    cv = load_cv()
    print(f"cv: {cv_scheme} {cv.n_splits}-fold, n_rows={cv.n_rows}")
    n_rows = len(frame)
    if n_rows != cv.n_rows:
        raise ValueError(
            f"train rows {n_rows} do not match cv n_rows {cv.n_rows}"
        )
    features = _xgb_features(
        prepare_features(
            frame,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        ),
        categorical_columns=categorical_columns,
    )
    categories = category_levels(
        features, categorical_columns=categorical_columns
    )
    target = frame[TARGET_COLUMN].to_numpy()
    y = target.astype(np.int64)
    if not np.all(target == y):
        raise ValueError("is_fraud is not integer-valued")
    record_ids = frame[ID_COLUMN].to_numpy()
    id_index = pd.Index(record_ids)
    n_folds = cv.n_splits
    write_config(
        path=config_path,
        payload={
            "exp_name": exp_name,
            "model": MODEL_NAME,
            "dataset_version": (
                {
                    "name": dataset_version.manifest.name,
                    "drop_columns": list(dataset_version.manifest.drop_columns),
                    "data_sha256": dataset_version.manifest.data_sha256,
                }
                if dataset_version is not None
                else None
            ),
            "args": model_args,
            "cv": {
                "scheme": cv_scheme,
                "n_splits": cv.n_splits,
                "random_state": cv.random_state,
                "shuffle": cv.shuffle,
            },
            "columns": {
                "id": ID_COLUMN,
                "target": TARGET_COLUMN,
                "features": list(feature_columns),
                "categorical": list(categorical_columns),
                "numeric": list(numeric_columns),
            },
            "categories": categories,
        },
    )
    print(f"wrote config: {_repo_relative(config_path)}")

    template = XGBClassifier(**model_args)
    oof_prob = np.full(n_rows, np.nan, dtype=np.float64)
    oof_fold = np.full(n_rows, -1, dtype=np.int64)
    fitted: list[object | None] = [None] * n_folds
    t_cv = time.perf_counter()
    for fold_id in tqdm(range(n_folds), desc="cv folds", total=n_folds):
        fold = cv.fold(fold_id)
        train_pos = _positions(
            id_index, ids=fold.train_ids, fold_id=fold_id, side="train"
        )
        val_pos = _positions(
            id_index, ids=fold.val_ids, fold_id=fold_id, side="val"
        )
        t_fold = time.perf_counter()
        model = clone(template)
        model.fit(features.iloc[train_pos], y[train_pos])
        proba = model.predict_proba(features.iloc[val_pos])[:, 1]
        oof_prob[val_pos] = proba
        oof_fold[val_pos] = fold_id
        fitted[fold_id] = model
        model_path = get_training_fold_model_path(exp_name, fold_id=fold_id)
        joblib.dump(model, model_path)
        print(
            f"fold {fold_id}: n_train={len(train_pos)} n_val={len(val_pos)} "
            f"in {time.perf_counter() - t_fold:.1f}s "
            f"model={_repo_relative(model_path)}"
        )
    if np.isnan(oof_prob).any() or np.any(oof_fold < 0):
        raise ValueError("OOF assignment did not cover every training row")
    models = [model for model in fitted if model is not None]
    if len(models) != n_folds:
        raise ValueError(f"expected {n_folds} fitted models, got {len(models)}")
    oof = pd.DataFrame(
        {
            OOF_ID_COL: record_ids,
            OOF_FOLD_COL: oof_fold,
            OOF_TARGET_COL: y,
            OOF_PROB_COL: oof_prob,
        }
    )
    write_oof(path=oof_path, frame=oof)
    print(f"wrote oof: {_repo_relative(oof_path)}")
    print(f"cv in {time.perf_counter() - t_cv:.1f}s")

    test_path = get_test_csv_path()
    print(f"test_csv: {_repo_relative(test_path)}")
    test_frame = pd.read_csv(test_path)
    if dataset_version is not None:
        groups_path = dataset_version.dir / RISK_CONTEXT_GROUPS_FILENAME
        if groups_path.exists():
            test_frame = _apply_risk_context_to_test(test_frame, groups_path)
        if PROFILE_CONTEXT_SIZE_COLUMN in feature_columns:
            train_raw = pd.read_csv(get_train_csv_path())
            test_frame = apply_profile_context_to_test(test_frame, train_raw)
        if any(name.startswith("grid_") for name in feature_columns):
            test_frame = apply_grid_slots_to_test(
                test_frame,
                include_exact=any(
                    name.startswith("grid_value_") for name in feature_columns
                ),
            )
        slot_z_path = dataset_version.dir / SLOT_Z_STATS_FILENAME
        if slot_z_path.exists():
            test_frame = apply_slot_z_to_test(test_frame, slot_z_path)
        slot_refs_path = dataset_version.dir / SLOT_REFS_FILENAME
        if slot_refs_path.exists():
            test_frame = apply_slot_refs_to_test(test_frame, slot_refs_path)
        anchor_freq_path = dataset_version.dir / ANCHOR_FREQ_FILENAME
        if anchor_freq_path.exists():
            test_frame = apply_anchor_freq_to_test(test_frame, anchor_freq_path)
        anchor_rate_path = dataset_version.dir / ANCHOR_RATE_FILENAME
        if anchor_rate_path.exists():
            test_frame = apply_anchor_rate_to_test(test_frame, anchor_rate_path)
        anchor_v2_path = dataset_version.dir / ANCHOR_V2_RATES_FILENAME
        if anchor_v2_path.exists():
            test_frame = apply_anchor_v2_rates_to_test(test_frame, anchor_v2_path)
        anchor_tilt_path = dataset_version.dir / ANCHOR_TILT_FILENAME
        if anchor_tilt_path.exists():
            test_frame = apply_anchor_tilt_to_test(test_frame, anchor_tilt_path)
        anchor_combined_path = dataset_version.dir / ANCHOR_COMBINED_FILENAME
        if anchor_combined_path.exists():
            test_frame = apply_anchor_combined_to_test(test_frame, anchor_combined_path)
        anchor_v3_path = dataset_version.dir / ANCHOR_V3_RATES_FILENAME
        if anchor_v3_path.exists():
            test_frame = apply_anchor_v3_rates_to_test(test_frame, anchor_v3_path)
        ic_rates_path = dataset_version.dir / IC_RATES_FILENAME
        if ic_rates_path.exists():
            test_frame = apply_ic_rates_to_test(test_frame, ic_rates_path)
        ic_cellz_path = dataset_version.dir / IC_CELLZ_FILENAME
        if ic_cellz_path.exists():
            test_frame = apply_cellz_to_test(test_frame, ic_cellz_path)
        ic_riskz_fine_path = dataset_version.dir / IC_RISKZ_FINE_FILENAME
        if ic_riskz_fine_path.exists():
            test_frame = apply_riskz_fine_to_test(test_frame, ic_riskz_fine_path)
        if IS_ALARM_COLUMN in feature_columns:
            test_frame = attach_is_alarm(test_frame)
        if N_ALARM_FLAGS_COLUMN in feature_columns:
            test_frame = attach_n_alarm_flags(test_frame)
        test_frame = test_frame.loc[:, [ID_COLUMN] + list(feature_columns)]
    # note: no frozen categories here - the config stores the post-cast
    # string levels, and feeding them back into prepare_features would map
    # boolean columns onto string levels (NaN codes). The two-step cast
    # reproduces the training levels deterministically instead.
    test_features = _xgb_features(
        prepare_features(
            test_frame,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        ),
        categorical_columns=categorical_columns,
    )
    for column in categorical_columns:
        train_levels = pd.Categorical(features[column]).categories
        if column.startswith("grid_"):
            # grid columns legitimately contain levels unseen in train (new
            # anchor values / slots appear in test). Re-anchor the codes onto
            # the training levels so unseen values become missing; those rows
            # still carry the raw numeric column.
            test_features[column] = pd.Categorical(
                test_features[column], categories=train_levels
            )
            continue
        test_levels = pd.Categorical(test_features[column]).categories
        if not test_levels.equals(train_levels):
            raise ValueError(
                f"test category levels for {column!r} differ from train: "
                f"{list(test_levels)} vs {list(train_levels)}"
            )
    n_test = len(test_features)
    test_prob = np.zeros(n_test, dtype=np.float64)
    t_sub = time.perf_counter()
    for fold_id in tqdm(range(n_folds), desc="test folds", total=n_folds):
        model = models[fold_id]
        test_prob += model.predict_proba(test_features)[:, 1]
    test_prob /= n_folds
    submission = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: test_prob,
        }
    )
    submission_path = get_training_submission_path(exp_name)
    submission.to_csv(submission_path, index=False)
    print(
        f"wrote submission: {_repo_relative(submission_path)} "
        f"n={n_test} in {time.perf_counter() - t_sub:.1f}s"
    )
    print("done")


def _xgb_features(
    features: pd.DataFrame, *, categorical_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Cast non-string category levels (booleans) to strings.

    XGBoost only accepts string or integer category values; pandas boolean
    categories are mapped to their string form. Applied identically to train
    and test, so the category codes stay aligned.
    """
    out = features.copy()
    for name in categorical_columns:
        if out[name].dtype != "category":
            continue
        levels = out[name].cat.categories
        if all(isinstance(level, str) for level in levels):
            continue
        out[name] = out[name].astype(str).astype("category")
    return out


def _positions(
    id_index: pd.Index,
    *,
    ids: tuple[int, ...],
    fold_id: int,
    side: str,
) -> np.ndarray:
    positions = id_index.get_indexer(ids)
    if np.any(positions < 0):
        raise KeyError(f"fold {fold_id} {side} contains ids missing from train.csv")
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

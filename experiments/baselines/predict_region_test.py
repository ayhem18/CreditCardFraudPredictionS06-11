"""Predict no-alarm test rows with a trained region-model experiment.

Loads the five fold models of ``--exp-name`` (a ``rgn_*`` run of
``train_region_variants.py``), rebuilds its variant model matrix for the
no-alarm rows of test.csv (categories frozen to the training levels,
within-profile quantiles measured against the full no-alarm train), and
writes ``test_no_alarm.csv`` (id, y_prob) next to the experiment's oof.

Usage::

    python experiments/baselines/predict_region_test.py --exp-name rgn_v4_quantiles_xgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.file_utils import (
    get_repo_root,
    get_test_csv_path,
    get_train_csv_path,
    get_training_config_path,
)
from ccf_ml.region_modeling import (
    add_device_group,
    add_merchant_tier,
    add_profile_id,
    build_no_alarm_frame,
    variant_by_name,
)
from ccf_ml.region_modeling.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    add_profile_quantiles_vs_train,
)
from ccf_ml.region_modeling.constants import (
    PROFILE_ID_COLUMN,
    REGION_ALWAYS_DROP,
    QUANTILE_NUMERIC_COLUMNS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()
    exp_name = args.exp_name
    config = json.loads(get_training_config_path(exp_name).read_text(encoding="utf-8"))
    variant = variant_by_name(config["variant"]["name"])
    models_dir = (
        get_repo_root() / "storage" / "artifacts" / "training" / exp_name / "models"
    )

    train_region = build_no_alarm_frame(pd.read_csv(get_train_csv_path()))
    test_frame = pd.read_csv(get_test_csv_path())
    has_alarm = alarm_flag_mask(test_frame, columns=ALARM_FLAG_COLUMNS)
    test_region = test_frame.loc[~has_alarm].copy()
    add_merchant_tier(test_region)
    add_device_group(test_region)
    add_profile_id(test_region)

    # variant feature columns, mirroring variant_model_matrix
    excluded = set(variant.drop_columns) | set(REGION_ALWAYS_DROP) | {
        ID_COLUMN,
        TARGET_COLUMN,
        "merchant_tier",
        "device_group",
        PROFILE_ID_COLUMN,
    }
    feature_columns = [
        column
        for column in train_region.columns
        if column not in excluded and not column.startswith("pq_")
    ]
    if variant.add_profile_quantiles:
        add_profile_quantiles_vs_train(train_region, test_region)
        feature_columns.extend(QUANTILE_PREFIX + c for c in QUANTILE_NUMERIC_COLUMNS)
    if variant.add_profile_id:
        feature_columns.append(PROFILE_ID_COLUMN)
    categorical_columns = [
        column
        for column in feature_columns
        if column in CATEGORICAL_FEATURE_COLUMNS or column == PROFILE_ID_COLUMN
    ]

    matrix = test_region.loc[:, [ID_COLUMN] + feature_columns].copy()
    for column in categorical_columns:
        levels = pd.Categorical(train_region[column].astype(str)).categories
        matrix[column] = pd.Categorical(matrix[column].astype(str), categories=levels)

    prob = np.zeros(len(matrix), dtype=np.float64)
    n_folds = config["cv"]["n_splits"]
    for fold_id in range(n_folds):
        model = joblib.load(models_dir / f"fold_{fold_id}.joblib")
        prob += model.predict_proba(matrix.drop(columns=[ID_COLUMN]))[:, 1]
    prob /= n_folds
    out = pd.DataFrame({ID_COLUMN: matrix[ID_COLUMN].to_numpy(), "y_prob": prob})
    out_path = models_dir.parent / "test_no_alarm.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} n={len(out)} mean_prob={float(prob.mean()):.6f}")


from ccf_ml.region_modeling.constants import QUANTILE_PREFIX  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

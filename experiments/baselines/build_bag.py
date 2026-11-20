"""Assemble a probability bag from member experiments; score its OOF.

Equal-weight probability mean of the members' OOF tables and of their
test submissions. All members must share the same frozen cv scheme and
the same per-row fold assignment - checked here, recorded in the bag's
config so downstream tools (``compare_regimes.py``) accept the bag like
any trained experiment.

Usage::

    python experiments/baselines/build_bag.py --exp-name xgbU_bagged_v1 \
        --member xgbU_tuned_v1 --member xgbU_seed_7 --member xgbU_seed_13 \
        --member xgbU_seed_21 --member xgbU_seed_2026
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.file_utils import (
    claim_training_exp_dir,
    get_repo_root,
    get_training_config_path,
    get_training_oof_path,
    get_training_submission_path,
)
from training_io import (
    OOF_FOLD_COL,
    OOF_ID_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    read_config,
    read_oof,
    write_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Bag experiment name to create. Must not already exist.",
    )
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="Member experiment name (repeatable). Order is recorded.",
    )
    args = parser.parse_args()
    members = list(args.member)
    if len(members) < 2:
        raise ValueError("a bag needs at least two members")

    cv_scheme: str | None = None
    oof: pd.DataFrame | None = None
    for member in members:
        config = read_config(path=get_training_config_path(member))
        scheme = config.get("cv", {}).get("scheme")
        if scheme is None:
            raise ValueError(f"{member} config has no cv scheme")
        if cv_scheme is None:
            cv_scheme = str(scheme)
        elif scheme != cv_scheme:
            raise ValueError(
                f"{member} is on cv scheme {scheme!r}; the bag so far is "
                f"{cv_scheme!r} - members must share one scheme"
            )
        part = read_oof(path=get_training_oof_path(member))
        if oof is None:
            oof = part.rename(columns={OOF_PROB_COL: member})
            continue
        merged = oof.merge(
            part.rename(columns={OOF_PROB_COL: member}),
            on=[OOF_ID_COL, OOF_FOLD_COL, OOF_TARGET_COL],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(oof):
            raise ValueError(
                f"{member} oof rows/folds/targets do not align with the bag"
            )
        oof = merged
    assert oof is not None and cv_scheme is not None

    y = oof[OOF_TARGET_COL].to_numpy(dtype=np.int64)
    fold = oof[OOF_FOLD_COL].to_numpy()
    bag_prob = oof.loc[:, members].mean(axis=1).to_numpy(dtype=np.float64)
    pooled = float(roc_auc_score(y, bag_prob))
    print(f"bag: {args.exp_name} = probability mean of {members}")
    print(f"oof auc: {pooled:.5f}")
    for fold_id in np.sort(np.unique(fold)):
        sel = fold == fold_id
        print(f"  fold {fold_id}: {roc_auc_score(y[sel], bag_prob[sel]):.5f}")
    for member in members:
        print(f"  member {member}: {roc_auc_score(y, oof[member].to_numpy()):.5f}")

    sub: pd.DataFrame | None = None
    for member in members:
        part = pd.read_csv(get_training_submission_path(member))
        part = part.rename(columns={TARGET_COLUMN: member})
        if sub is None:
            sub = part
            continue
        sub = sub.merge(part, on=ID_COLUMN, how="inner", validate="one_to_one")
        if len(sub) != len(part):
            raise ValueError(f"{member} submission rows do not align with the bag")
    assert sub is not None
    bag_sub = sub.loc[:, members].mean(axis=1).to_numpy(dtype=np.float64)

    exp_dir = claim_training_exp_dir(args.exp_name)
    submission_path = exp_dir / "submission.csv"
    pd.DataFrame(
        {ID_COLUMN: sub[ID_COLUMN].to_numpy(), TARGET_COLUMN: bag_sub}
    ).to_csv(submission_path, index=False)
    write_config(
        path=get_training_config_path(args.exp_name),
        payload={
            "exp_name": args.exp_name,
            "model": "xgb_probability_bag",
            "bag": members,
            "cv": {
                "scheme": cv_scheme,
                "n_splits": int(len(np.unique(fold))),
            },
            "oof_auc": pooled,
        },
    )
    oof_out = exp_dir / "oof.csv"
    pd.DataFrame(
        {
            OOF_ID_COL: oof[OOF_ID_COL].to_numpy(),
            OOF_FOLD_COL: fold,
            OOF_TARGET_COL: y,
            OOF_PROB_COL: bag_prob,
        }
    ).to_csv(oof_out, index=False)
    print(f"wrote {submission_path.relative_to(get_repo_root())} n={len(sub)}")
    print(f"wrote {oof_out.relative_to(get_repo_root())}")
    print(f"wrote {get_training_config_path(args.exp_name).relative_to(get_repo_root())}")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

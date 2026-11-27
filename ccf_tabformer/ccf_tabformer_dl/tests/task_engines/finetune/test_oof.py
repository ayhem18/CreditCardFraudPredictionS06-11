"""OOF concatenation and mean-of-folds test scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccf_tabformer_dl.task_engines.finetune.oof import (
    STACK_OOF_AUC,
    assemble_oof,
    average_test_scores,
    write_oof_csv,
    write_submission_csv,
)


def _write_fold(
    root: Path,
    *,
    fold: int,
    val_ids: list[int],
    val_labels: list[float],
    val_logits: list[float],
    test_ids: list[int],
    test_logits: list[float],
) -> Path:
    folder = root / f"fold_{fold}"
    folder.mkdir()
    np.save(folder / "val_row_ids.npy", np.array(val_ids, dtype=np.int64))
    np.save(folder / "val_labels.npy", np.array(val_labels, dtype=np.float32))
    np.save(folder / "val_logits.npy", np.array(val_logits, dtype=np.float32))
    np.save(folder / "test_row_ids.npy", np.array(test_ids, dtype=np.int64))
    np.save(folder / "test_logits.npy", np.array(test_logits, dtype=np.float32))
    return folder


def test_assemble_oof_and_mean_test(tmp_path: Path) -> None:
    test_ids = [100, 101]
    fold0 = _write_fold(
        tmp_path,
        fold=0,
        val_ids=[1, 2],
        val_labels=[0.0, 1.0],
        val_logits=[-4.0, 4.0],
        test_ids=test_ids,
        test_logits=[0.0, 2.0],
    )
    fold1 = _write_fold(
        tmp_path,
        fold=1,
        val_ids=[3, 4],
        val_labels=[1.0, 0.0],
        val_logits=[4.0, -4.0],
        test_ids=test_ids,
        test_logits=[2.0, 0.0],
    )
    assembled = assemble_oof(fold_payloads=(fold0, fold1))
    assert assembled["n_oof"] == 4
    assert assembled["n_folds"] == 2
    assert assembled["oof_auc"] == pytest.approx(1.0)
    assert assembled["stack_oof_auc"] == STACK_OOF_AUC
    assert assembled["row_ids"].tolist() == [1, 2, 3, 4]
    row_ids, scores = average_test_scores(fold_payloads=(fold0, fold1))
    assert row_ids.tolist() == test_ids
    expected = 0.5 * (
        (1.0 / (1.0 + np.exp(-np.array([0.0, 2.0]))))
        + (1.0 / (1.0 + np.exp(-np.array([2.0, 0.0]))))
    )
    np.testing.assert_allclose(scores, expected)
    oof_csv = tmp_path / "oof.csv"
    sub_csv = tmp_path / "submission.csv"
    write_oof_csv(assembled, dest=oof_csv)
    write_submission_csv(row_ids=row_ids, scores=scores, dest=sub_csv)
    oof_text = oof_csv.read_text(encoding="utf-8")
    assert oof_text.splitlines()[0] == "id,fold_id,is_fraud,y_prob"
    assert sub_csv.read_text(encoding="utf-8").splitlines()[0] == "id,is_fraud"


def test_assemble_oof_rejects_overlapping_ids(tmp_path: Path) -> None:
    fold0 = _write_fold(
        tmp_path,
        fold=0,
        val_ids=[1],
        val_labels=[0.0],
        val_logits=[-1.0],
        test_ids=[9],
        test_logits=[0.0],
    )
    fold1 = _write_fold(
        tmp_path,
        fold=1,
        val_ids=[1],
        val_labels=[1.0],
        val_logits=[1.0],
        test_ids=[9],
        test_logits=[0.0],
    )
    with pytest.raises(ValueError, match="overlap"):
        assemble_oof(fold_payloads=(fold0, fold1))

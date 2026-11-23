"""Run P1–P9 census, vs-parent shift, and compactness on a region frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from ccf_ml.data_analysis.census import CensusRow, census_as_dict, census_row, region_totals
from ccf_ml.data_analysis.compactness import (
    compactness_1d,
    compactness_joint,
    midrank_quantiles,
    write_pair_arrays,
)
from ccf_ml.data_analysis.constants import (
    AMOUNT_COLUMN,
    CENSUS_FILENAME,
    COMPACT_MIN_FRAUD,
    COMPACT_MIN_NONFRAUD,
    COMPACTNESS_CELL_FILENAME,
    COMPACTNESS_FRAUD_FILENAME,
    DISTANCE_COLUMN,
    NOTEBOOK_STEM,
    PAIRS_DIRNAME,
    PARENT_SHIFT_FILENAME,
    SHUFFLE_SEED,
)
from ccf_ml.data_analysis.parent_shift import (
    ShiftRow,
    mark_distilled,
    measure_shift,
    shift_as_dict,
)
from ccf_ml.data_analysis.plots import plot_compactness_1d, plot_compactness_joint
from ccf_ml.data_analysis.profiles import (
    Cell,
    ProfileSpec,
    RegionView,
    catalog,
    iter_cells,
    load_no_alarm_train,
    path_slug,
    region_view_from_frame,
)
from ccf_ml.file_utils import get_script_plot_dir, get_script_table_dir


class ProfileTables(NamedTuple):
    """Summary tables written under the notebook stem."""

    census: pd.DataFrame
    parent_shift: pd.DataFrame
    compactness_cell: pd.DataFrame
    compactness_fraud: pd.DataFrame


def tables_dir(*, script_name: str = NOTEBOOK_STEM) -> Path:
    """``storage/artifacts/<script_name>/tables``."""
    return get_script_table_dir(script_name)


def pairs_root(*, script_name: str = NOTEBOOK_STEM) -> Path:
    """``tables/pairs``."""
    return tables_dir(script_name=script_name) / PAIRS_DIRNAME


def artifacts_exist(*, script_name: str = NOTEBOOK_STEM) -> bool:
    """True when the four summary files from a previous run are on disk."""
    directory = tables_dir(script_name=script_name)
    names = (
        CENSUS_FILENAME,
        PARENT_SHIFT_FILENAME,
        COMPACTNESS_CELL_FILENAME,
        COMPACTNESS_FRAUD_FILENAME,
    )
    n_names = len(names)
    for i in range(n_names):
        if not (directory / names[i]).is_file():
            return False
    return True


def load_tables(*, script_name: str = NOTEBOOK_STEM) -> ProfileTables:
    """Read the four summary files from a previous run."""
    directory = tables_dir(script_name=script_name)
    census = pd.read_csv(directory / CENSUS_FILENAME)
    parent_shift = pd.read_csv(directory / PARENT_SHIFT_FILENAME)
    compactness_cell = pd.read_csv(directory / COMPACTNESS_CELL_FILENAME)
    compactness_fraud = pd.read_parquet(directory / COMPACTNESS_FRAUD_FILENAME)
    return ProfileTables(
        census=census,
        parent_shift=parent_shift,
        compactness_cell=compactness_cell,
        compactness_fraud=compactness_fraud,
    )


def run_cat_profiling(
    *,
    frame: pd.DataFrame | None = None,
    specs: tuple[ProfileSpec, ...] | None = None,
    script_name: str = NOTEBOOK_STEM,
    force: bool = False,
    make_plots: bool = True,
    seed: int = SHUFFLE_SEED,
) -> ProfileTables:
    """Census, vs-parent, compactness, and pair arrays for every catalog spec.

    Reuses existing summary files unless ``force`` is true.
    """
    if artifacts_exist(script_name=script_name) and not force:
        return load_tables(script_name=script_name)
    plt.switch_backend("Agg")
    if frame is None:
        frame = load_no_alarm_train()
    if specs is None:
        specs = catalog()
    view = region_view_from_frame(frame)
    totals = region_totals(y=view.y)
    rng = np.random.default_rng(seed)
    table_dir = tables_dir(script_name=script_name)
    pair_dir = pairs_root(script_name=script_name)
    pair_dir.mkdir(parents=True, exist_ok=True)
    census_records: list[dict[str, object]] = []
    shift_records: list[dict[str, object]] = []
    cell_records: list[dict[str, object]] = []
    fraud_frames: list[pd.DataFrame] = []
    n_specs = len(specs)
    for spec_index in range(n_specs):
        spec = specs[spec_index]
        cells = list(iter_cells(view, spec))
        print(
            f"[{spec_index + 1}/{n_specs}] {spec.spec_name} cells={len(cells)}",
            flush=True,
        )
        spec_shifts: list[ShiftRow] = []
        for cell in tqdm(cells, desc=spec.spec_name, leave=False):
            y_cell = view.y[cell.cell_index]
            counted = census_row(
                profile_id=cell.profile_id,
                spec_name=cell.spec_name,
                cell_key=cell.cell_key,
                parent_key=cell.parent_key,
                y_cell=y_cell,
                totals=totals,
            )
            census_records.append(census_as_dict(counted))
            n_numeric = len(spec.numerics)
            for numeric_index in range(n_numeric):
                numeric = spec.numerics[numeric_index]
                spec_shifts.extend(
                    _shifts_for_numeric(
                        view,
                        spec,
                        cell,
                        numeric=numeric,
                        rng=rng,
                    )
                )
                compact = _compactness_for_numeric(
                    view,
                    spec,
                    cell,
                    counted=counted,
                    numeric=numeric,
                    pair_dir=pair_dir,
                    script_name=script_name,
                    make_plots=make_plots,
                )
                if compact is None:
                    continue
                cell_records.append(compact[0])
                fraud_frames.append(compact[1])
            joint = _compactness_for_joint(
                view,
                spec,
                cell,
                counted=counted,
                pair_dir=pair_dir,
                script_name=script_name,
                make_plots=make_plots,
            )
            if joint is not None:
                cell_records.append(joint[0])
                fraud_frames.append(joint[1])
        marked = mark_distilled(spec_shifts)
        n_marked = len(marked)
        for i in range(n_marked):
            shift_records.append(shift_as_dict(marked[i]))
    census = pd.DataFrame(census_records)
    parent_shift = pd.DataFrame(shift_records)
    compactness_cell = pd.DataFrame(cell_records)
    if fraud_frames:
        compactness_fraud = pd.concat(fraud_frames, ignore_index=True)
    else:
        compactness_fraud = _empty_fraud_table()
    census.to_csv(table_dir / CENSUS_FILENAME, index=False)
    parent_shift.to_csv(table_dir / PARENT_SHIFT_FILENAME, index=False)
    compactness_cell.to_csv(table_dir / COMPACTNESS_CELL_FILENAME, index=False)
    compactness_fraud.to_parquet(table_dir / COMPACTNESS_FRAUD_FILENAME, index=False)
    _write_summary(
        table_dir / "summary.json",
        totals_n=totals.n,
        totals_n_fraud=totals.n_fraud,
        census=census,
        parent_shift=parent_shift,
        compactness_cell=compactness_cell,
    )
    return ProfileTables(
        census=census,
        parent_shift=parent_shift,
        compactness_cell=compactness_cell,
        compactness_fraud=compactness_fraud,
    )


def _shifts_for_numeric(
    view: RegionView,
    spec: ProfileSpec,
    cell: Cell,
    *,
    numeric: str,
    rng: np.random.Generator,
) -> list[ShiftRow]:
    cell_x = view.numerics[numeric][cell.cell_index]
    n_parents = len(cell.parents)
    rows = [None] * n_parents
    axis_values = None
    if spec.window_axis is not None:
        axis_values = view.numerics[spec.window_axis][cell.parents[0].index]
    for i in range(n_parents):
        parent = cell.parents[i]
        parent_x = view.numerics[numeric][parent.index]
        rows[i] = measure_shift(
            cell_x=cell_x,
            parent_x=parent_x,
            profile_id=spec.profile_id,
            spec_name=spec.spec_name,
            cell_key=cell.cell_key,
            parent_key=parent.key,
            parent_axis=parent.axis,
            numeric=numeric,
            require_scramble=spec.scramble,
            hit_verdict=spec.hit_verdict,
            window=cell.window,
            parent_axis_values=axis_values,
            rng=rng if spec.scramble else None,
        )
    return rows


def _compact_occupancy(*, y_cell: np.ndarray) -> bool:
    n_fraud = int(np.sum(y_cell == 1))
    n_nonfraud = int(y_cell.size) - n_fraud
    return n_fraud >= COMPACT_MIN_FRAUD and n_nonfraud >= COMPACT_MIN_NONFRAUD


def _compactness_for_numeric(
    view: RegionView,
    spec: ProfileSpec,
    cell: Cell,
    *,
    counted: CensusRow,
    numeric: str,
    pair_dir: Path,
    script_name: str,
    make_plots: bool,
) -> tuple[dict[str, object], pd.DataFrame] | None:
    y_cell = view.y[cell.cell_index]
    if not _compact_occupancy(y_cell=y_cell):
        return None
    values = view.numerics[numeric][cell.cell_index]
    ids = view.ids[cell.cell_index]
    per_fraud, rollup, extras = compactness_1d(values=values, y=y_cell, ids=ids)
    destination = (
        pair_dir / path_slug(spec.spec_name) / path_slug(cell.cell_key) / path_slug(numeric)
    )
    write_pair_arrays(
        destination,
        extras=extras,
        write_d_ij=cell.window is None,
    )
    if make_plots and _should_plot_1d(spec, counted=counted, numeric=numeric):
        plot_dir = get_script_plot_dir(script_name, section=_plot_section(spec, counted))
        plot_compactness_1d(
            q=per_fraud["q"],
            nn5=per_fraud["nn5_abs_mean"],
            title=f"{spec.spec_name} | {cell.cell_key} | {numeric}",
            destination=plot_dir / f"{path_slug(cell.cell_key)}__{path_slug(numeric)}.png",
        )
    cell_record = _cell_rollup_record(
        spec,
        cell,
        numeric=numeric,
        counted=counted,
        rollup=rollup,
    )
    fraud_table = _fraud_table(spec, cell, numeric=numeric, per_fraud=per_fraud)
    return cell_record, fraud_table


def _compactness_for_joint(
    view: RegionView,
    spec: ProfileSpec,
    cell: Cell,
    *,
    counted: CensusRow,
    pair_dir: Path,
    script_name: str,
    make_plots: bool,
) -> tuple[dict[str, object], pd.DataFrame] | None:
    axes = spec.joint_coordinates
    if len(axes) < 2:
        return None
    y_cell = view.y[cell.cell_index]
    if not _compact_occupancy(y_cell=y_cell):
        return None
    n_axes = len(axes)
    n_rows = int(cell.cell_index.size)
    matrix = np.empty((n_rows, n_axes), dtype=np.float64)
    for j in range(n_axes):
        matrix[:, j] = midrank_quantiles(view.numerics[axes[j]][cell.cell_index])
    ids = view.ids[cell.cell_index]
    per_fraud, rollup, extras = compactness_joint(
        quantile_matrix=matrix,
        y=y_cell,
        ids=ids,
    )
    numeric_name = "joint:" + "+".join(axes)
    destination = (
        pair_dir
        / path_slug(spec.spec_name)
        / path_slug(cell.cell_key)
        / path_slug(numeric_name)
    )
    write_pair_arrays(
        destination,
        extras=extras,
        write_d_ij=cell.window is None,
    )
    if make_plots and _should_plot_joint(spec, counted=counted):
        plot_dir = get_script_plot_dir(script_name, section=_plot_section(spec, counted))
        plot_compactness_joint(
            nn5=per_fraud["nn5_abs_mean"],
            far5=per_fraud["far5_abs_mean"],
            title=f"{spec.spec_name} | {cell.cell_key} | {numeric_name}",
            destination=plot_dir / f"{path_slug(cell.cell_key)}__joint.png",
        )
    cell_record = _cell_rollup_record(
        spec,
        cell,
        numeric=numeric_name,
        counted=counted,
        rollup=rollup,
    )
    fraud_table = _fraud_table(spec, cell, numeric=numeric_name, per_fraud=per_fraud)
    return cell_record, fraud_table


def _should_plot_1d(
    spec: ProfileSpec,
    *,
    counted: CensusRow,
    numeric: str,
) -> bool:
    if spec.require_plots and numeric == DISTANCE_COLUMN:
        return True
    if counted.tag == "bulky_cold" and numeric in {DISTANCE_COLUMN, AMOUNT_COLUMN}:
        return True
    return False


def _should_plot_joint(spec: ProfileSpec, *, counted: CensusRow) -> bool:
    if spec.require_plots:
        return True
    return counted.tag == "bulky_cold"


def _plot_section(spec: ProfileSpec, counted: CensusRow) -> str:
    if spec.require_plots:
        return "p6"
    if counted.tag == "bulky_cold":
        return "bulky_cold"
    return "other"


def _cell_rollup_record(
    spec: ProfileSpec,
    cell: Cell,
    *,
    numeric: str,
    counted: CensusRow,
    rollup: dict[str, float],
) -> dict[str, object]:
    record: dict[str, object] = {
        "profile_id": spec.profile_id,
        "spec_name": spec.spec_name,
        "cell_key": cell.cell_key,
        "parent_key": cell.parent_key,
        "numeric": numeric,
        "tag": counted.tag,
    }
    for key, value in rollup.items():
        record[key] = value
    return record


def _fraud_table(
    spec: ProfileSpec,
    cell: Cell,
    *,
    numeric: str,
    per_fraud: dict[str, np.ndarray],
) -> pd.DataFrame:
    table = pd.DataFrame(per_fraud)
    table.insert(0, "numeric", numeric)
    table.insert(0, "cell_key", cell.cell_key)
    table.insert(0, "spec_name", spec.spec_name)
    table.insert(0, "profile_id", spec.profile_id)
    return table


def _empty_fraud_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "profile_id",
            "spec_name",
            "cell_key",
            "numeric",
            "id",
            "q",
            "min_d",
            "max_d",
            "nn5_abs_mean",
            "far5_abs_mean",
            "n_doubles_0_01",
            "n_doubles_0_05",
        ]
    )


def _write_summary(
    path: Path,
    *,
    totals_n: int,
    totals_n_fraud: int,
    census: pd.DataFrame,
    parent_shift: pd.DataFrame,
    compactness_cell: pd.DataFrame,
) -> None:
    n_p6 = 0
    if not census.empty:
        n_p6 = int((census["profile_id"] == "P6").sum())
    n_bulky = 0
    n_hot = 0
    if not census.empty:
        n_bulky = int((census["tag"] == "bulky_cold").sum())
        n_hot = int((census["tag"] == "hot").sum())
    n_fe = 0
    n_distilled = 0
    if not parent_shift.empty:
        n_fe = int((parent_shift["verdict"] == "fe_candidate").sum())
        n_distilled = int(parent_shift["distilled"].astype(bool).sum())
    payload = {
        "n": totals_n,
        "n_fraud": totals_n_fraud,
        "n_census_cells": int(len(census)),
        "n_p6_cells": n_p6,
        "n_bulky_cold": n_bulky,
        "n_hot": n_hot,
        "n_fe_candidate": n_fe,
        "n_distilled": n_distilled,
        "n_compactness_cells": int(len(compactness_cell)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

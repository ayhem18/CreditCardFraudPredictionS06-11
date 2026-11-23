"""P1–P9 catalog, cells, and the no-alarm region view."""

from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import alarm_flag_mask, as_binary_label, as_record_ids
from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.data_analysis.constants import (
    AGE_COLUMN,
    AMOUNT_COLUMN,
    AUTH_COLUMN,
    CARD_COLUMN,
    CATEGORICAL_COLUMNS,
    CELL_JOIN,
    CHANNEL_COLUMN,
    DEVICE_COLUMN,
    DISTANCE_COLUMN,
    EXTRA_NUMERICS,
    FEATURED_NUMERICS,
    HOUR_COLUMN,
    JOINT_COORDINATES,
    MERCHANT_COLUMN,
    NEW_MERCHANT_COLUMN,
    RISK_COLUMN,
    VELOCITY_COLUMN,
)
from ccf_ml.data_analysis.windows import (
    Window,
    age_membership,
    age_windows,
    hour_membership,
    hour_windows,
)
from ccf_ml.file_utils import get_train_csv_path


class ParentSlice(NamedTuple):
    """One nested reference for measurement 2."""

    axis: str
    key: str
    index: np.ndarray


class Cell(NamedTuple):
    """Observed rows of one profile value, with nested parent slices."""

    profile_id: str
    spec_name: str
    cell_key: str
    parent_key: str
    cell_index: np.ndarray
    parents: tuple[ParentSlice, ...]
    window: Window | None


class ProfileSpec(NamedTuple):
    """One catalog entry: how to form cells and which numerics to measure."""

    profile_id: str
    spec_name: str
    cell_columns: tuple[str, ...]
    parent_columns: tuple[str, ...]
    extra_parent_columns: tuple[str, ...]
    numerics: tuple[str, ...]
    joint_coordinates: tuple[str, ...]
    window_axis: str | None
    scramble: bool
    hit_verdict: str
    require_plots: bool


class RegionView(NamedTuple):
    """Aligned arrays for the no-alarm region."""

    frame: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray
    numerics: dict[str, np.ndarray]
    n: int


def load_no_alarm_train() -> pd.DataFrame:
    """Train rows with every alarm flag off. Does not add ``merchant_tier``."""
    frame = pd.read_csv(get_train_csv_path())
    has_alarm = alarm_flag_mask(frame)
    region = frame.loc[~has_alarm].reset_index(drop=True)
    return region


def region_view_from_frame(frame: pd.DataFrame) -> RegionView:
    """Pull id, label, and numeric columns into aligned arrays."""
    y = as_binary_label(frame[TARGET_COLUMN], column=TARGET_COLUMN)
    ids = as_record_ids(frame[ID_COLUMN], column=ID_COLUMN)
    needed = set(FEATURED_NUMERICS + EXTRA_NUMERICS + (HOUR_COLUMN, AGE_COLUMN))
    numerics: dict[str, np.ndarray] = {}
    for name in needed:
        if name not in frame.columns:
            continue
        numerics[name] = frame[name].to_numpy(dtype=np.float64)
    return RegionView(
        frame=frame,
        y=y,
        ids=ids,
        numerics=numerics,
        n=int(len(frame)),
    )


def joined_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Join column values with ``CELL_JOIN``. Empty columns → ``all``."""
    n_columns = len(columns)
    n_rows = int(len(frame))
    if n_columns == 0:
        return np.full(n_rows, "all", dtype=object)
    if n_columns == 1:
        return frame[columns[0]].map(str).to_numpy()
    return frame[list(columns)].astype(str).agg(CELL_JOIN.join, axis=1).to_numpy()


def grouped_indices(keys: np.ndarray) -> dict[str, np.ndarray]:
    """Map each distinct key to integer positions, first-seen order."""
    grouped = pd.Series(keys).groupby(keys, sort=False)
    result: dict[str, np.ndarray] = {}
    for key, positions in grouped.groups.items():
        result[str(key)] = np.asarray(positions, dtype=np.int64)
    return result


def _with_joint(spec: ProfileSpec) -> ProfileSpec:
    """Fill ``joint_coordinates`` from the shared numeric axes minus profile axes."""
    return spec._replace(joint_coordinates=joint_coordinates_for(spec))


def joint_coordinates_for(spec: ProfileSpec) -> tuple[str, ...]:
    """Drop a joint axis when it is a profile axis of this spec."""
    n_axes = len(JOINT_COORDINATES)
    kept = [None] * n_axes
    filled = 0
    for i in range(n_axes):
        name = JOINT_COORDINATES[i]
        if spec.window_axis == name:
            continue
        if name in spec.cell_columns:
            continue
        kept[filled] = name
        filled += 1
    if filled < 2:
        return ()
    return tuple(kept[:filled])


def catalog() -> tuple[ProfileSpec, ...]:
    """P1–P9 in plan order. ``channel × device_type`` lives only on P4."""
    p1_numerics = FEATURED_NUMERICS + EXTRA_NUMERICS
    p4_numerics = (
        AMOUNT_COLUMN,
        VELOCITY_COLUMN,
        DISTANCE_COLUMN,
        AGE_COLUMN,
    )
    p2_numerics = (AMOUNT_COLUMN, VELOCITY_COLUMN, RISK_COLUMN)
    p3_numerics = (AMOUNT_COLUMN, RISK_COLUMN)
    p7_numerics = (AMOUNT_COLUMN, VELOCITY_COLUMN)
    p9_numerics = (DISTANCE_COLUMN, AMOUNT_COLUMN)
    p4_others = (
        MERCHANT_COLUMN,
        NEW_MERCHANT_COLUMN,
        AUTH_COLUMN,
        CARD_COLUMN,
        CHANNEL_COLUMN,
    )
    p5_others = (
        MERCHANT_COLUMN,
        NEW_MERCHANT_COLUMN,
        AUTH_COLUMN,
        CARD_COLUMN,
    )
    n_p1 = 1 + len(CATEGORICAL_COLUMNS)
    n_p4 = len(p4_others)
    n_p5 = len(p5_others)
    n_total = n_p1 + 2 + n_p4 + n_p5 + 4
    specs = [None] * n_total
    index = 0
    specs[index] = ProfileSpec(
        profile_id="P1",
        spec_name="P1:all",
        cell_columns=(),
        parent_columns=(),
        extra_parent_columns=(),
        numerics=p1_numerics,
        joint_coordinates=(),
        window_axis=None,
        scramble=False,
        hit_verdict="descriptive_only",
        require_plots=False,
    )
    index += 1
    for column in CATEGORICAL_COLUMNS:
        specs[index] = ProfileSpec(
            profile_id="P1",
            spec_name=f"P1:{column}",
            cell_columns=(column,),
            parent_columns=(),
            extra_parent_columns=(),
            numerics=p1_numerics,
            joint_coordinates=(),
            window_axis=None,
            scramble=False,
            hit_verdict="descriptive_only",
            require_plots=False,
        )
        index += 1
    specs[index] = ProfileSpec(
        profile_id="P2",
        spec_name="P2:merchant_category|is_new_merchant",
        cell_columns=(MERCHANT_COLUMN, NEW_MERCHANT_COLUMN),
        parent_columns=(MERCHANT_COLUMN,),
        extra_parent_columns=(),
        numerics=p2_numerics,
        joint_coordinates=(),
        window_axis=None,
        scramble=False,
        hit_verdict="fe_candidate",
        require_plots=False,
    )
    index += 1
    specs[index] = ProfileSpec(
        profile_id="P3",
        spec_name="P3:merchant_category|auth_method",
        cell_columns=(MERCHANT_COLUMN, AUTH_COLUMN),
        parent_columns=(MERCHANT_COLUMN,),
        extra_parent_columns=(),
        numerics=p3_numerics,
        joint_coordinates=(),
        window_axis=None,
        scramble=False,
        hit_verdict="fe_candidate",
        require_plots=False,
    )
    index += 1
    for other in p4_others:
        specs[index] = _with_joint(
            ProfileSpec(
                profile_id="P4",
                spec_name=f"P4:device_type|{other}",
                cell_columns=(DEVICE_COLUMN, other),
                parent_columns=(DEVICE_COLUMN,),
                extra_parent_columns=(),
                numerics=p4_numerics,
                joint_coordinates=(),
                window_axis=None,
                scramble=False,
                hit_verdict="fe_candidate",
                require_plots=False,
            )
        )
        index += 1
    for other in p5_others:
        specs[index] = _with_joint(
            ProfileSpec(
                profile_id="P5",
                spec_name=f"P5:channel|{other}",
                cell_columns=(CHANNEL_COLUMN, other),
                parent_columns=(CHANNEL_COLUMN,),
                extra_parent_columns=(),
                numerics=p4_numerics,
                joint_coordinates=(),
                window_axis=None,
                scramble=False,
                hit_verdict="fe_candidate",
                require_plots=False,
            )
        )
        index += 1
    specs[index] = _with_joint(
        ProfileSpec(
            profile_id="P6",
            spec_name="P6:device_type|channel",
            cell_columns=(DEVICE_COLUMN, CHANNEL_COLUMN),
            parent_columns=(DEVICE_COLUMN,),
            extra_parent_columns=(CHANNEL_COLUMN,),
            numerics=p4_numerics,
            joint_coordinates=(),
            window_axis=None,
            scramble=False,
            hit_verdict="fe_candidate",
            require_plots=True,
        )
    )
    index += 1
    specs[index] = ProfileSpec(
        profile_id="P7",
        spec_name="P7:merchant_category|hour_window",
        cell_columns=(MERCHANT_COLUMN,),
        parent_columns=(MERCHANT_COLUMN,),
        extra_parent_columns=(),
        numerics=p7_numerics,
        joint_coordinates=(),
        window_axis=HOUR_COLUMN,
        scramble=True,
        hit_verdict="fe_candidate",
        require_plots=False,
    )
    index += 1
    specs[index] = ProfileSpec(
        profile_id="P8",
        spec_name="P8:merchant_category|age_window",
        cell_columns=(MERCHANT_COLUMN,),
        parent_columns=(MERCHANT_COLUMN,),
        extra_parent_columns=(),
        numerics=p7_numerics,
        joint_coordinates=(),
        window_axis=AGE_COLUMN,
        scramble=True,
        hit_verdict="fe_candidate",
        require_plots=False,
    )
    index += 1
    specs[index] = _with_joint(
        ProfileSpec(
            profile_id="P9",
            spec_name="P9:device_type|age_window",
            cell_columns=(DEVICE_COLUMN,),
            parent_columns=(DEVICE_COLUMN,),
            extra_parent_columns=(),
            numerics=p9_numerics,
            joint_coordinates=(),
            window_axis=AGE_COLUMN,
            scramble=True,
            hit_verdict="fe_candidate",
            require_plots=False,
        )
    )
    index += 1
    if index != n_total:
        raise RuntimeError(f"catalog fill {index} != planned {n_total}")
    filled = tuple(spec for spec in specs if spec is not None)
    if len(filled) != n_total:
        raise RuntimeError("catalog has unfilled slots")
    return filled


def iter_cells(view: RegionView, spec: ProfileSpec) -> Iterator[Cell]:
    """Yield every observed cell of ``spec``. Missing combinations are absent."""
    frame = view.frame
    n = view.n
    region_index = np.arange(n, dtype=np.int64)
    if spec.window_axis is None:
        yield from _iter_categorical_cells(
            frame,
            spec,
            region_index=region_index,
        )
        return
    yield from _iter_window_cells(view, spec)


def _iter_categorical_cells(
    frame: pd.DataFrame,
    spec: ProfileSpec,
    *,
    region_index: np.ndarray,
) -> Iterator[Cell]:
    cell_keys = joined_keys(frame, spec.cell_columns)
    cell_groups = grouped_indices(cell_keys)
    if spec.parent_columns:
        parent_keys = joined_keys(frame, spec.parent_columns)
        parent_groups = grouped_indices(parent_keys)
    else:
        parent_keys = np.full(int(len(frame)), "all", dtype=object)
        parent_groups = {"all": region_index}
    extra_groups: dict[str, np.ndarray] | None = None
    extra_keys: np.ndarray | None = None
    if spec.extra_parent_columns:
        extra_keys = joined_keys(frame, spec.extra_parent_columns)
        extra_groups = grouped_indices(extra_keys)
    for cell_key, cell_index in cell_groups.items():
        parent_key = str(parent_keys[cell_index[0]])
        parent_index = parent_groups[parent_key]
        n_parents = 1 + int(extra_groups is not None and extra_keys is not None)
        parents = [None] * n_parents
        parents[0] = ParentSlice(
            axis=spec.parent_columns[0] if spec.parent_columns else "region",
            key=parent_key,
            index=parent_index,
        )
        if extra_groups is not None and extra_keys is not None:
            extra_key = str(extra_keys[cell_index[0]])
            extra_axis = spec.extra_parent_columns[0]
            parents[1] = ParentSlice(
                axis=extra_axis,
                key=extra_key,
                index=extra_groups[extra_key],
            )
        yield Cell(
            profile_id=spec.profile_id,
            spec_name=spec.spec_name,
            cell_key=str(cell_key),
            parent_key=parent_key,
            cell_index=cell_index,
            parents=tuple(parents),
            window=None,
        )


def _iter_window_cells(
    view: RegionView,
    spec: ProfileSpec,
) -> Iterator[Cell]:
    frame = view.frame
    parent_keys = joined_keys(frame, spec.cell_columns)
    parent_groups = grouped_indices(parent_keys)
    if spec.window_axis == HOUR_COLUMN:
        windows = hour_windows()
        axis_values = view.numerics[HOUR_COLUMN]
    elif spec.window_axis == AGE_COLUMN:
        ages = view.numerics[AGE_COLUMN]
        age_min = int(np.floor(np.nanmin(ages)))
        age_max = int(np.ceil(np.nanmax(ages)))
        windows = age_windows(age_min=age_min, age_max=age_max)
        axis_values = ages
    else:
        raise ValueError(f"unknown window axis {spec.window_axis!r}")
    n_windows = len(windows)
    for parent_key, parent_index in parent_groups.items():
        parent_axis = axis_values[parent_index]
        parent_slice = ParentSlice(
            axis=spec.cell_columns[0] if spec.cell_columns else "region",
            key=str(parent_key),
            index=parent_index,
        )
        for i in range(n_windows):
            window = windows[i]
            if window.kind == "hour":
                inside = hour_membership(parent_axis, window=window)
            else:
                inside = age_membership(parent_axis, window=window)
            if not np.any(inside):
                continue
            cell_index = parent_index[inside]
            yield Cell(
                profile_id=spec.profile_id,
                spec_name=spec.spec_name,
                cell_key=f"{parent_key}{CELL_JOIN}{window.kind}{window.label}",
                parent_key=str(parent_key),
                cell_index=cell_index,
                parents=(parent_slice,),
                window=window,
            )


def path_slug(text: str) -> str:
    """Filesystem-safe fragment for pair and plot directories."""
    n = len(text)
    chars = [None] * n
    for i in range(n):
        ch = text[i]
        if ch.isalnum() or ch in "._-":
            chars[i] = ch
        else:
            chars[i] = "_"
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    if not slug:
        return "cell"
    return slug[:180]

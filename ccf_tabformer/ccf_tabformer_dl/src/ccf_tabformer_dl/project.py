"""Resolve the ccf_tabformer ProjectLayout from any path inside the tree."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.project_layout.names import MARKER_FILENAME, TRACKING_BACKEND_FILENAME
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend


def load_project_layout(*, start: Path, tracking_backend: str | None) -> ProjectLayout:
    start_file = File.from_absolute(path=start)
    backend = (
        tracking_backend
        if tracking_backend is not None
        else read_project_tracking_backend(start=start_file)
    )
    marker = read_object(
        file=File.from_parent_walk(start=start_file, basename=MARKER_FILENAME)
    )
    name = marker["name"]
    if not isinstance(name, str) or not name:
        raise TypeError(f"marker name must be a non-empty str, got {name!r}")
    return ProjectLayout.from_start(
        start=start_file, project_name=name, tracking_backend=backend
    )


def read_project_tracking_backend(*, start: File) -> str:
    marker = File.from_parent_walk(start=start, basename=MARKER_FILENAME)
    backend_file = File.from_under(
        root=marker.parent(), relative_path=TRACKING_BACKEND_FILENAME
    )
    text = parse_non_empty_str(
        backend_file.path.read_text(encoding="utf-8").strip(), label="tracking_backend"
    )
    parsed = parse_one_of(
        text,
        allowed=TrackingBackend.BACKENDS,
        label="tracking_backend",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(parsed, str):
        raise TypeError(f"tracking_backend must be str, got {type(parsed)!r}")
    return parsed

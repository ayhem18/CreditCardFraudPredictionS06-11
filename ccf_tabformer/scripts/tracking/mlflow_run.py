"""Open MLflow UI on this project's tracking store.

Named mlflow_run.py so scripts/tracking/ does not shadow the mlflow package.

Store paths come from ProjectLayout (logging JSON).

  python scripts/tracking/mlflow_run.py
  python scripts/tracking/mlflow_run.py --port 5001
"""

from __future__ import annotations

import sys
from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.project_layout.serve import run_mlflow


if __name__ == "__main__":
    raise SystemExit(
        run_mlflow(
            start=File.from_absolute(path=Path(__file__).resolve()),
            argv=sys.argv[1:],
        )
    )

"""Tabular-BERT for Kaggle S6E11 — digit-sequence MFM pretraining + fine-tune.

See ``TABFORMER_PLAN_V2.md`` in the competition repo for the design. This
package holds the project-local task code (tokenized-table loader, model,
mask sampler, losses, metrics, FitSteps assembly) that wires into the
``dl_toolbox`` engines/runs/loggers.
"""

from __future__ import annotations

__version__ = "0.1.0"

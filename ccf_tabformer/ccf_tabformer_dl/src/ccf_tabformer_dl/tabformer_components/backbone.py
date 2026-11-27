"""Transfer a pretrained MFM backbone (embedder + encoder) into a ClassifierModel.

The full run saves an `MfmModel` checkpoint whose state_dict holds `embedder.*`,
`encoder.*` and `head.*` (the last is the reconstruction head, dropped here). This
loads the `embedder.*` / `encoder.*` tensors into a `ClassifierModel` — which
names those submodules identically — and leaves the fresh `classifier_head.*`
untouched.

Full-run Lightning checkpoints prefix every module key with ``net.``
(``TrainValLightningModule.net``). That prefix is stripped before matching.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from ccf_tabformer_dl.tabformer_components.classifier_model import ClassifierModel

BACKBONE_PREFIXES = ("embedder.", "encoder.")
LIGHTNING_NET_PREFIX = "net."
RECONSTRUCTION_HEAD_PREFIX = "head."
CLASSIFIER_HEAD_PREFIX = "classifier_head."


def load_backbone_from_checkpoint(
    model: ClassifierModel,
    *,
    checkpoint_path: Path,
    map_location: torch.device,
) -> None:
    """Load embedder/encoder weights in place from a full-run checkpoint."""
    if not isinstance(model, ClassifierModel):
        raise TypeError(f"model must be ClassifierModel, got {type(model)!r}")
    if not isinstance(checkpoint_path, Path):
        raise TypeError(
            f"checkpoint_path must be Path, got {type(checkpoint_path)!r}"
        )
    if not checkpoint_path.is_absolute():
        raise ValueError(
            f"checkpoint_path must be an absolute Path, got {checkpoint_path}"
        )
    if not isinstance(map_location, torch.device):
        raise TypeError(
            f"map_location must be torch.device, got {type(map_location)!r}"
        )
    payload = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            "checkpoint must be a dict with key 'state_dict', "
            f"got {type(payload)!r}"
        )
    state = payload["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"state_dict must be a mapping, got {type(state)!r}")
    stripped = _without_lightning_net_prefix(state)
    _require_backbone_keys(stripped)
    incompatible = model.load_state_dict(stripped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    _require_only_prefix(
        missing,
        prefix=CLASSIFIER_HEAD_PREFIX,
        label="missing_keys",
        require_non_empty=True,
    )
    _require_only_prefix(
        unexpected,
        prefix=RECONSTRUCTION_HEAD_PREFIX,
        label="unexpected_keys",
        require_non_empty=True,
    )


def _without_lightning_net_prefix(
    state: Mapping[str, object],
) -> dict[str, object]:
    keys = list(state.keys())
    n = len(keys)
    if n == 0:
        raise ValueError("checkpoint state_dict is empty")
    prefixed = 0
    for i in range(n):
        if str(keys[i]).startswith(LIGHTNING_NET_PREFIX):
            prefixed += 1
    if prefixed == 0:
        return dict(state)
    if prefixed != n:
        raise ValueError(
            "checkpoint state_dict mixes Lightning 'net.' keys with unprefixed keys"
        )
    stripped: dict[str, object] = {}
    n_prefix = len(LIGHTNING_NET_PREFIX)
    for i in range(n):
        stripped[str(keys[i])[n_prefix:]] = state[keys[i]]
    return stripped


def _require_backbone_keys(state: Mapping[str, object]) -> None:
    n_prefixes = len(BACKBONE_PREFIXES)
    counts = [0] * n_prefixes
    for key in state:
        for i in range(n_prefixes):
            if key.startswith(BACKBONE_PREFIXES[i]):
                counts[i] += 1
                break
    for i in range(n_prefixes):
        if counts[i] < 1:
            raise ValueError(
                "checkpoint has no embedder.* / encoder.* keys after stripping "
                f"net. (counts={counts})"
            )


def _require_only_prefix(
    keys: list[str],
    *,
    prefix: str,
    label: str,
    require_non_empty: bool,
) -> None:
    n = len(keys)
    if require_non_empty and n == 0:
        raise ValueError(f"{label} must be non-empty ({prefix}*)")
    for i in range(n):
        if not keys[i].startswith(prefix):
            raise ValueError(
                f"{label} must contain only {prefix}* keys, got {keys}"
            )

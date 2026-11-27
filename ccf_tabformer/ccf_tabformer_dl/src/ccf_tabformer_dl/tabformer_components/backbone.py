"""Transfer a pretrained MFM backbone (embedder + encoder) into a ClassifierModel.

The full run saves an `MfmModel` checkpoint whose state_dict holds `embedder.*`,
`encoder.*` and `head.*` (the last is the reconstruction head, dropped here). This
loads the `embedder.*` / `encoder.*` tensors into a `ClassifierModel` — which
names those submodules identically — and leaves the fresh `classifier_head.*`
untouched.
"""

from __future__ import annotations

from pathlib import Path

from ccf_tabformer_dl.tabformer_components.classifier_model import ClassifierModel

BACKBONE_PREFIXES = ("embedder.", "encoder.")


def load_backbone_from_checkpoint(
    model: ClassifierModel,
    *,
    checkpoint_path: Path,
    map_location: str = "cpu",
) -> None:
    """Load the backbone weights in place. Raise if no backbone key matched or if
    any `embedder.*` / `encoder.*` parameter of `model` is left unfilled.

    Contract (body deferred): `torch.load(checkpoint_path, map_location)`, take its
    `state_dict`, strip the LightningModule wrapper prefix (confirm the exact
    prefix against a real `full_run` checkpoint), keep only keys under
    `BACKBONE_PREFIXES`, `model.load_state_dict(filtered, strict=False)`, then
    assert `missing_keys` contains *only* `classifier_head.*` and `unexpected_keys`
    contains *only* the reconstruction `head.*`.
    """
    raise NotImplementedError(
        "contract stub: filter the checkpoint state_dict to embedder./encoder. "
        "and load_state_dict(strict=False); verify the backbone matched fully"
    )

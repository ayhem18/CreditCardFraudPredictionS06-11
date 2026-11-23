# Mistakes of `xgbU_tuned_av2` — plan

Chain: observation → hypothesis → data → tests → (FE/model only if
indicated). One folder per model: `experiments/eda/<model>_oof_analysis/`.

## Observation

The champion (OOF AUC 0.94703, unified folds) still misorders fraud–clean
pairs. The generator composes multiplicatively (anchors × numeric
partner = product of marginals; anchors × categorical partner = constant
lifts), and the model already receives every ingredient — so its
remaining mistakes are either (a) rows the model class genuinely cannot
distinguish, or (b) rows where it had the information and still ranked
wrong.

## Hypothesis

The mistake pile splits into an irreducible floor (frauds whose
leaf-twins are clean: the model cannot tell them apart) and a small set
of true misses whose profile may name a pocket or a capacity limit.

## Data

OOF predictions of `xgbU_tuned_av2` (every row scored by the fold model
that never saw it), the fold models themselves (leaf assignments via
`apply`), and the anchor_v2 feature frame used in training.

## Tests

- **M1 inventory**: buried frauds (OOF p below 0.02 / 0.01 — the global
  rate is 0.0167, so below base rate = worse than a blind guess); cleans
  ranked above each fraud (AUC is lost on pairs).
- **M2 where ranking fails**: per-region OOF AUC (alarm region,
  merchant_category, auth_method, risk/velocity blocks) and the split of
  misordered pairs into within-region vs cross-region.
- **M3 leaf-twin arbiter**: for each buried fraud, the clean rows sharing
  the most leaves of its own fold model (400 trees; share = co-assigned
  fraction). Best twin share ≥ 0.9 → at the model's floor; no such twin →
  true miss. Chance baseline reported (mean leaf occupancy).
- **M4 miss profile**: true misses vs well-ranked frauds across features,
  readable units (shares, medians).

## FE / model

Only if M4 names something concrete.

## Files

- notebook: `mistakes_v1.ipynb`
- plots: `storage/artifacts/xgbU_tuned_av2_oof_analysis/mistakes_v1/`

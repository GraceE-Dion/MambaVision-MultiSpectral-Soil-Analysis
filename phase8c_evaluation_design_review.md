# Phase 8C — Composite Evaluation Design (Locked)

**Status:** Dataset generation is complete and verified. Finding 1 closed, Finding 2 closed, generator frozen, 1760 composites generated across 203 targets, mechanical integrity check passed clean (0 ROI failures, 0 A/B/C rule violations, 0 non-admissible repairs). Two review passes are now fully incorporated, including the final wording fix and the AP-computation implementation requirement below. This document is the fully locked spec for `13q_phase8c_composite_evaluation.py`. No further methodology review is needed before coding; the next review should be a code/protocol verification or dry-run result.

**New script:** `13q_phase8c_composite_evaluation.py`. `phase8b_evaluate.py` is left untouched — it served Phase 8B's own purpose and is not repurposed or deleted.

## Scope

Four frozen models, unchanged since Phase 8B:
1. MambaVision_S full-image classifier
2. YOLOv8s — pretrained (8B.1)
3. YOLOv8s — scratch (8B.2)
4. Mamba-YOLO-T — scratch (8B.3)

Each is run against:
- The 1760 generated composites (conditions A/B/C)
- A **matched clean baseline**: each model's performance on the original, uncomposited target images. **Clean inference is run once per unique target per model** (203 targets), not duplicated per condition — condition A's 199-target eligible subset and B/C's 203-target subset are both derived downstream from this single clean-prediction set by selection, not by re-running inference.

## Per-composite/per-model output: raw, not pre-aggregated

Every model run against every composite writes one row of raw prediction-level output — not a pre-aggregated per-condition summary. Locked from the original design review:
1. **Bootstrap resampling for 8E must happen at the image/target level with full metric recomputation per replicate**, not by resampling already-computed per-image AP values.
2. **Donors are not averaged together.** Each (target, donor, condition, model) combination is its own row.

### Continuous detector output (per row)

- `matched_pred_class`, `matched_confidence`, `matched_bbox`, `matched_iou`
- `localized_iou50` (bool), `localized_iou75` (bool)
- `num_detections` (total surviving detections after confidence/NMS filtering, before matching)

**Deterministic target-matching rule:** apply the same confidence threshold and NMS settings as Phase 8B, then among the surviving predictions select the one with the **highest IoU against the target GT box** — never the highest-confidence box irrespective of location. No overlap → unlocalized miss (`matched_iou = 0`, both `localized_iou*` false, `matched_pred_class = null`).

### AP implementation requirement (not optional)

A single per-image maximum-IoU matched prediction is **sufficient for localization retention, conditional/unconditional classification, and donor attraction, but not for mAP50/mAP50–95**, which require the complete post-NMS detection set per image (including false positives) and confidence ranking. `13q` therefore does both, not one or the other:

1. **mAP50 / mAP50–95 are computed directly through the same Ultralytics evaluation machinery used in 8B**, run once per condition (clean, A, B, C) per detector, against that condition's image set — not reconstructed from the per-row matched prediction.
2. **All post-NMS detections per image are additionally persisted** (not just the matched one) in a companion raw-detections file, so AP is independently reconstructable later if needed and nothing about the detector's actual output is discarded.

The per-composite raw row's `matched_*` fields remain the single derived prediction used for every other metric below; they are not used to compute AP.

### Classifier output (per row)

Predicted class, confidence, and — where readily available from the forward pass — the **full 11-class probability/logit vector**, retained for later ordinal/confidence analysis in 8E.

## Pass-through experimental identity and repair-quality metadata

Every row carries the full experimental identity, copied from `composite_metadata.json` by `composite_id` (already unique and mechanically verified by the Finding-2 integrity check):

`target_id`, `target_class`, `target_source`, `target_modality`, `donor_id`, `donor_class`, `donor_source`, `donor_modality`, `condition`, `donor_rep`, `same_modality`, `repair_tier`, `repair_method`, `repair_candidate_count`, `repair_candidate_support`, `weak_candidate_support`, `repair_selected_score`, `repair_shrink_fraction_used`.

Pure pass-through — never used here for filtering, weighting, or routing (Finding 2's closed gate). Exists so 8E can slice results without a second join later.

## Metrics computed and reported (not gated behind formal CIs — those are 8E's job)

| Outcome | Classifier | Detectors |
|---|---|---|
| Unconditional classification accuracy | ✓ | — |
| End-to-end correct detection/classification | — | ✓ |
| Conditional classification accuracy (given successful localization) | N/A | ✓ |
| Localization retention @ IoU ≥ 0.50 | N/A | ✓ primary |
| Localization retention @ IoU ≥ 0.75 | N/A | ✓ secondary |
| Continuous matched IoU | N/A | ✓ |
| mAP50 | N/A | ✓ |
| mAP50–95 | N/A | ✓ |
| Conditional donor attraction (B/C) | ✓ | ✓ |
| Unconditional donor attraction (B/C) | ✓ | ✓ |
| Prediction confidence | ✓ | ✓ |

1. **Localization retention** (detectors only): among target GT ROIs successfully localized under clean conditions, the proportion still localized under the perturbed (A/B/C) condition, at IoU ≥ 0.50 (primary) and IoU ≥ 0.75 (secondary).

2. **Detector AP, reported as mAP50 and mAP50–95**, matching Phase 8B's own locked detector evaluation settings exactly, per condition and clean baseline, computed as described above.

3. **Conditional classification accuracy** (detectors): among successfully localized targets, gated at IoU ≥ 0.50, consistent with the primary localization-retention threshold — the proportion assigned the correct moisture class.

4. **Unconditional classification accuracy** (classifier): accuracy over 100% of composites — no localization step to gate on. Reported alongside detector numbers with an explicit conditioning caveat; never renormalized to force comparability.

5. **End-to-end correct detection/classification rate** (detectors): proportion of **all eligible composite observations** for which the detector both (a) localizes the target at IoU ≥ 0.50 and (b) assigns the correct class. Gives detectors and the classifier a genuine common denominator alongside metric 4. All three (unconditional classifier accuracy, detector end-to-end rate, detector conditional accuracy) are retained.

6. **Donor attraction** (B/C only):
   - **Conditional**: among misclassified-but-localized (detectors) or misclassified (classifier) targets, frequency that the predicted class equals the donor's class. Numerator and denominator are always saved alongside the ratio.
     $$DA_{conditional} = \frac{N(\text{localized, misclassified, predicted donor class})}{N(\text{localized, misclassified})}$$
   - **Unconditional**: proportion of *all eligible* perturbation observations for which the predicted class equals the donor class; a detector localization failure counts as not donor-attracted, not as excluded from the denominator.
     $$DA_{unconditional} = \frac{N(\text{predicted donor class})}{N(\text{all eligible perturbation observations})}$$

7. **Prediction confidence** (both model types): raw per-row confidence (and classifier probability vector where available) retained for 8E confidence-calibration analysis.

Statistical framing:
- Condition A reported alongside B and C — **not** subtracted as a noise floor.
- Predefined contrasts:
  $$\Delta_A = M_A - M_{clean,A}, \quad \Delta_B = M_B - M_{clean,B}, \quad \Delta_C = M_C - M_{clean,C}$$
  and **B − A**.
- **B − A common-subset restriction**: computed as $M_{B,\,common199} - M_{A,\,common199}$, restricted to the intersection of eligible targets (A's 199). **Condition B's own descriptive result is still reported in full on all 203 targets** — only the paired B − A contrast is restricted to the common 199.
- No exclusion or accuracy-equivalence claims without the formal bootstrap CIs — computed in 8E from this script's raw output.

## Batch/runtime structure

One full evaluation pass per model (matches the two conda environments: `mambavision` for the classifier, `mambayolo` for the three detectors). Incremental checkpointing: raw prediction rows appended to `results/phase8c/eval_raw/{model_name}.jsonl` as produced; the matched-clean-baseline pass is its own checkpointed sub-run.

**Resume-key safeguard**: a row is "already done" only when both `model_id` **and** `composite_id` match an existing row **and** the pass-through experimental-identity fields (or a hash of them) also match what's on disk — not `composite_id` alone. Protects against silently resuming against a since-changed or different-version dataset.

## Explicit non-goals for this script

- No confidence intervals (8E).
- No cross-donor averaging.
- No exclusion of any composite based on repair-quality metadata (Finding 2 closed gate).
- No noise-floor subtraction of Condition A from B/C.
- No renormalization of the classifier's unconditional accuracy to match the detectors' conditional basis (the end-to-end rate already provides the common-denominator comparison).
- No single-threshold "AP75" metric distinct from mAP50–95; IoU 0.75 is used only for localization retention.

## Status

**Evaluation architecture: APPROVED — fully locked.** `13q_phase8c_composite_evaluation.py` implements this spec exactly, including the AP-via-Ultralytics-machinery requirement and the full post-NMS detection persistence. See the script itself, delivered alongside this document.

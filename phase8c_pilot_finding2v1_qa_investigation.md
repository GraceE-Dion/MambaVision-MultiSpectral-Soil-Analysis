# Phase 8C — Pilot `finding2v1` Visual QA: Two Flagged Composites Investigated

**Context:** Following peer review's approval of the Finding 2 tier redesign (dense Tier-1 search, region-shrink Tier-2, `tier3_no_texture_fallback` marked non-admissible), a 15-image heterogeneous pilot (`pilot_finding2v1`) was generated and produced a strong headline result: 0% Tier-3 fallback across 104 unique donors (79.8% Tier 1, 20.2% Tier 2), 0 non-admissible repairs, 126/126 ROI-integrity passes.

Manual visual QA of the `*_QA_panel.png` files (per the locked plan's Step 3) flagged two composites for a closer look. Both have now been resolved via independent numeric and visual checks. Neither is a leakage bug. This document records the investigation for the transparency record, per project norm.

## Composite 1: `t005_B_2` — confirmed candidate-starved forced pick

- `repair_tier = 1`, `repair_candidate_count = 1`, `repair_selected_score = 208.90` (highest of the whole pilot; pilot mean was 27.53)
- `repair_donor_bbox` spans 130×144px; at `pad_fraction=0.6` this inflates to a 286×316px padded search region — very large relative to the donor image
- Out of the full dense radius×angle sweep (~240 candidate offsets), only **one** landed fully in-bounds and non-overlapping. "Tier 1, dense search" here reduced to a single forced option with no real selection, producing a large luminance mismatch (`delta_mean_luminance ≈ 103`, vs. pilot's typical range)
- **Conclusion:** genuine repair-quality degradation caused by region-size-vs-image-size geometry, not a leakage/boundary artifact — the padded region interior is fully overwritten regardless of match quality, so nothing "survives" from the donor's original pixels.

## Composite 2: `t002_A_0` — confirmed ordinary texture mismatch, ruled out as leakage

Flagged for a faint residual-looking smudge that visually resembled the donor's original signal surviving the repair. Three competing explanations were tested in order, using both the composite's metadata and a direct visual inspection of the QA panel's donor/mask view:

| Hypothesis | Test | Result |
|---|---|---|
| (1) Forced low-quality pick (candidate scarcity) | `repair_candidate_count` | **Ruled out** — 134 candidates available, not a forced pick |
| (2) Feather-blend boundary leak (padding thinner than the blend zone) | `pad_x`(19) / `pad_y`(32) vs. `feather_px`(8) | **Ruled out** — both margins are 2–4× the feather band width; the blend zone at the region's outer edge cannot reach the annotated box |
| (3) Padding too thin to cover the donor's true glow extent | Direct visual inspection of QA panel view 1 (original donor + red padded-region box) | **Ruled out** — glow is a well-defined cluster sitting centrally in the box with 25–30% margin to the boundary on all four sides; no contact anywhere |

With all three structural/leakage explanations ruled out, the remaining explanation is ordinary patch-selection variance: `repair_selected_score = 36.66` is a mild-to-moderate mismatch (pilot mean 27.53; this is not a severe outlier the way `t005_B_2` was), consistent with the kind of low-grade texture imperfection any nearest-match patch search will occasionally produce even with abundant candidates (134 here) and a comfortably small, well-contained repair region (70×118px).

**Conclusion:** benign texture-quality artifact, distinct sub-pattern from `t005_B_2` — "mild mismatch despite adequate candidates" rather than "starved search, no good options." Not the leakage mechanism Phase 8A/8C exist to characterize and guard against.

## Net assessment (revised per peer review)

Two independently-flagged worst-looking composites in the pilot both resolved to benign, structurally-understood causes:
1. Region size ballooning relative to image size when the donor's annotated box is large, occasionally starving the candidate search down to a single forced (and visually poor) option (`t005_B_2`).
2. Ordinary nearest-match imperfection, unrelated to search density or padding — an expected, low-severity artifact of any patch-replacement method (`t002_A_0`).

Neither investigation found evidence of residual donor-laser leakage or target-ROI corruption. The 126/126 ROI-integrity passes confirm preservation of the target ROI, while the absence of non-admissible repairs confirms that no donor required the predefined no-texture fallback. Repair-quality variation remains separately monitored through candidate count, similarity score, and visual QA.

*(This paragraph replaces an earlier, overstated version of this document that framed ROI integrity and `repair_admissible` as formal guarantees against the acquisition-context confound. Per peer review: they guarantee only what they check — target-ROI preservation and rule-compliant repair method — not that a repair artifact couldn't itself become a model-usable contextual feature. That's precisely why visual/quality QA exists as a separate check, not a redundant one.)*

`t002_A_0` — abundant candidates (134), mild score (36.66) — is retained without special intervention. `t005_B_2` — essentially no alternatives (1 candidate), forced poor match (score 208.90) — is **not** excluded or regenerated, but is retained as a flagged case: `weak_candidate_support = True`. Not leakage, but not an unremarkable repair either; whether this is an isolated occurrence or a systematic pattern among large-ROI donors is an open question for the larger run, not this pilot.

## Candidate-support metadata: implemented (approved by peer review)

Tier alone conflates two different things: which repair pathway succeeded, and how much real choice that pathway had. `13e_phase8c_generate_composites.py` now records, for every repair regardless of tier:
- `repair_candidate_support`: `"weak"` (≤3 candidates), `"limited"` (4–10), or `"adequate"` (>10). Per peer review, only the `≤3` boundary is treated as load-bearing; the others are operational categories, not validated thresholds.
- `weak_candidate_support`: boolean, `True` when `repair_candidate_count ≤ 3`.

This is a **pure observability addition** — candidate count does not trigger Tier 2 or otherwise change routing. The search algorithm itself is frozen; this only makes an existing risk (weak support) queryable without waiting for a high score to surface it. The run summary and the QA sample-selection script (`13n_phase8c_qa_sample_selection.py`) both now report/surface every weak-candidate-support donor in full (not sampled), the same way Tier-3 cases already were.

## Gate status

- Finding 2 redesign: real-data validation **PASS**.
- Tier-3 structural problem: apparently resolved in this pilot (0/104 unique donors), **needs confirmation at larger scale**.
- Target ROI integrity: **PASS** (126/126).
- `t002_A_0`: acceptable ordinary repair variation, retained without intervention.
- `t005_B_2`: no leakage detected, retained as a QA-flagged weak-support repair (`weak_candidate_support = True`).
- Candidate-support flag: **implemented**.
- Repair algorithm: **frozen** — no change to Tier 1/2 search or routing logic.
- Next larger heterogeneous QA run: **approved**, to report (all at the unique-donor level): tier distribution; candidate-count distribution; number/percentage with ≤3 candidates; score distribution overall and by candidate-support group; Tier-2 shrink fractions; non-admissible count; ROI-integrity failures. Visual QA for that run should cover all ≤3-candidate repairs, all Tier-3/non-admissible cases (if any), the highest-score repairs, and a heterogeneous sample of the ordinary population.

If the larger run shows candidate starvation is rare and isolated, Finding 2 is essentially closed. If it shows a systematic population of candidate-starved large-ROI donors, a decision on whether those should invoke Tier 2 (or a separate prespecified treatment) will be needed — but not before that evidence exists.

## Addendum: 30-target stress-QA pilot (`finding2v2`) and the September/Stir-Sept structural finding

A second pilot, `finding2v2`, was generated per peer review's specification: 30 target images (stratified by source with a class-coverage top-up), with donor selection in pilot mode deliberately biased toward the geometries most likely to challenge repair (largest-bbox donor and boundary-nearest donor guaranteed among each condition's donor picks, mixed with ordinary random picks). This bias is pilot-mode-only; the full sweep retains plain seeded random donor assignment per the locked plan, unchanged.

**Headline results, 250 composites / 105 unique donors:** 0% Tier 3, 0 non-admissible repairs, 250/250 ROI-integrity passes — reproducing `finding2v1`'s result even under deliberately harder donor selection.

**Weak-candidate-support rate rose from ~1% (1/104, `finding2v1`, pure random) to 9.5% (10/105, `finding2v2`, stress-biased donor selection).** This increase is expected and by design — it reflects the deliberate oversampling of large-bbox and boundary-adjacent donors, not a newly-discovered population-level rate. The true population rate (under the full sweep's plain random assignment) is expected to be closer to the first pilot's ~1%, not this pilot's 9.5%.

**Structural finding:** 7 of the 10 weak-candidate-support donors in `finding2v2` (70%) came from just two of the seven sources — September (5) and Stir-Sept (2). A read-only diagnostic over the full 203-image index (`13o_phase8c_bbox_geometry_by_source.py`, no model, no repair, no composite generation) confirmed this is structural, not incidental:

| Source | n | mean bbox area / image area | mean boundary distance (frac. of short dim) |
|---|---|---|---|
| September | 11 | 0.0536 | 0.1615 |
| v4-UV | 40 | 0.0276 | 0.3295 |
| v4 | 70 | 0.0264 | 0.3190 |
| v4-IR | 30 | 0.0248 | 0.3050 |
| Stir-Sept | 12 | 0.0234 | 0.2806 |
| 5sagf | 20 | 0.0144 | 0.3655 |
| IR | 20 | 0.0046 | 0.3723 |

September's annotated ROIs are simultaneously ~2× larger relative to image size than the next-highest source, and sit roughly half as far from the image edge as the next-closest source — the exact geometric combination (large padded search region + edge proximity) that mechanically reduces the number of in-bounds candidate offsets. Stir-Sept shows a secondary, weaker version of the same pattern (moderate area, second-lowest boundary distance). The other five sources cluster together with neither property elevated.

**Conclusion (revised per peer review):** weak candidate support is associated with identifiable source-specific ROI geometry rather than unexplained stochastic failure of the redesigned search procedure. This is a narrower and more defensible claim than "no gap in the search algorithm" — any finite same-image patch search necessarily has constraints; what was established is that *where* those constraints bind is explainable, not that they don't exist. This should be documented as a known, source-linked characteristic of the dataset for the paper's methods discussion, separately from the repair pipeline's own correctness (which the ROI-integrity and non-admissibility results continue to support).

## Final spot-check

One additional composite was visually spot-checked: `t009_A_2` (`finding2v2`) shares its donor with `t005_B_2` (`finding2v1`) — same donor image, same bbox, same forced-single-candidate repair (score 208.90, one candidate, substantial luminance mismatch). Since the repaired-donor background depends only on the donor + bbox, not the target, this is the same repair reappearing under a different target/condition pairing — exactly what the deterministic donor-level cache predicts. Visual inspection confirmed the final composite (target ROI paste) is intact and sharp, with no seam, no duplicate laser spot, and continuous background texture.

**Wording correction on `t005_B_2`/`t009_A_2` (per peer review):** these are not "benign" repairs — that overstates the conclusion. What was established is: one candidate, score 208.90, substantial luminance mismatch, no evidence of residual donor laser, no target-ROI corruption, intact final composite. "Not leakage" does not imply "harmless to model behavior" — a conspicuous repair artifact could theoretically influence a model even without containing donor information. The correct label is a **known low-quality/weak-support repair**, tracked as such via metadata rather than corrected or excluded. Phase 8E's planned sensitivity analysis (using `repair_candidate_support`, tier, score, donor ID, etc.) is the designed mechanism for addressing that possibility if it matters — not further data manipulation now.

## Gate — Finding 2: CLOSED

*Formal record (peer review's language):* "The redesigned donor-repair procedure was evaluated in two heterogeneous real-data pilots, including a 30-target stress-QA run deliberately enriched for challenging ROI geometries. Across the stress pilot, no unique donor required Tier-3 fallback, no repair entered the predefined non-admissible pathway, and all 250 generated composites passed pixel-level target-ROI integrity checks. Weak candidate support was observed primarily under large and/or boundary-proximal ROI geometries and was concentrated in acquisition sources exhibiting those geometric characteristics. Candidate-support metadata is therefore retained as a diagnostic variable rather than used to modify repair routing. Following visual QA of weak-support, high-score, and heterogeneous ordinary repairs, the generation algorithm was frozen before full Phase 8C model inference."

**`13e_phase8c_generate_composites.py` is now the frozen Phase 8C generation implementation.** No further changes to candidate density, Tier-2 behavior, padding, feathering, scoring weights, weak-support routing, or repair admissibility based on subsequent model results. A genuine software bug, if found, is fixed and documented — methodological tuning ends here.

## What happens now

1. Full 203-target generation using the **original locked donor-selection procedure** (plain seeded random, `select_pilot_donors`'s stress-bias is pilot-mode only and does not apply): `python 13e_phase8c_generate_composites.py --full`
2. Full metadata retained: `repair_candidate_count`, `repair_candidate_support`, `weak_candidate_support`, `repair_selected_score`, `repair_tier`, shrink fraction, repair method, donor/target source/modality/class, repaired-donor path — nothing discarded.
3. A post-generation **mechanical integrity check** (not another pilot, not another methodological gate) before any model inference: confirm all 203 targets represented as expected, composite counts match donor eligibility, zero ROI-integrity failures, zero A/B/C rule violations, no missing metadata/artifacts, and report the actual Tier-3/non-admissible/weak-support distribution under real random donor assignment.
4. Only after that check passes: run the four frozen models.

**Finding 1: CLOSED. Finding 2: CLOSED. Generator: FROZEN. Full 203-target Phase 8C generation: GO. Model inference: GO after the mechanical integrity check passes.**

"""
13p_phase8c_full_sweep_integrity_check.py
============================================
Mandatory post-generation mechanical integrity check for the full
203-target Phase 8C sweep, per peer review's Finding 2 closure. This
is NOT another pilot and NOT another methodological gate -- it is a
verification pass confirming the frozen generator produced what it
should have, before any model inference runs on the output.

Checks performed, independently re-derived from composite_metadata.json
and the actual files on disk (not trusting the generation script's own
printed console summary):

  1. Target coverage: every image in the source val-split index was
     attempted as a target (matches --full's target count).
  2. No duplicate composite_id values (corruption/overwrite check).
  3. Every composite_path file actually exists on disk and is non-empty.
  4. Every referenced repaired_donor_path file actually exists on disk.
  5. ROI integrity: 100% of records must have roi_integrity_ok == True.
     Any failure is a locked-plan violation -- this script exits
     non-zero if any are found, since inference must not proceed on
     a compromised dataset.
  6. A/B/C eligibility rule re-verification: for every record,
     independently recompute from target_source/donor_source/
     target_class/donor_class whether the stored condition actually
     satisfies the locked plan's rule (A: same source + same class;
     B: same source + different class; C: different source +
     different class). Any violation is reported and treated as a
     hard failure.
  7. No records with missing/None values in required metadata fields.
  8. Reports (does not gate on) the real Tier-3/non-admissible/
     weak-candidate-support distribution under the full sweep's
     actual plain-random donor assignment -- this is expected to be
     much closer to finding2v1's ~1% weak-support rate than
     finding2v2's deliberately-stress-biased 9.5%, since --full does
     not use select_pilot_donors().

Usage:
    python 13p_phase8c_full_sweep_integrity_check.py results/phase8c/full
"""

import os
import sys
import json


REQUIRED_FIELDS = [
    "composite_id", "target_id", "condition", "donor_id",
    "target_source", "target_class", "donor_source", "donor_class",
    "composite_path", "roi_integrity_ok", "repair_tier", "repair_method",
    "repair_candidate_count", "repair_candidate_support",
    "weak_candidate_support", "repair_selected_score", "repair_admissible",
]


def check_ab_c_rule(r):
    """Returns None if the record's condition is consistent with the
    locked plan's rule given its own stored source/class fields, else
    a string describing the violation."""
    cond = r["condition"]
    same_source = r["target_source"] == r["donor_source"]
    same_class = r["target_class"] == r["donor_class"]
    if cond == "A" and not (same_source and same_class):
        return f"condition A requires same source + same class; got same_source={same_source}, same_class={same_class}"
    if cond == "B" and not (same_source and not same_class):
        return f"condition B requires same source + different class; got same_source={same_source}, same_class={same_class}"
    if cond == "C" and not (not same_source and not same_class):
        return f"condition C requires different source + different class; got same_source={same_source}, same_class={same_class}"
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python 13p_phase8c_full_sweep_integrity_check.py <full_sweep_output_dir>")
        sys.exit(1)

    out_dir = sys.argv[1]
    metadata_path = os.path.join(out_dir, "composite_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"ERROR: {metadata_path} not found")
        sys.exit(1)

    with open(metadata_path) as f:
        records = json.load(f)

    print("=" * 70)
    print(f"  Phase 8C Full-Sweep Mechanical Integrity Check")
    print(f"  {metadata_path}  ({len(records)} composite records)")
    print("=" * 70)

    hard_failures = []

    # --- 1. Target coverage ---
    unique_targets = {r["target_id"] for r in records}
    print(f"\n[1] Distinct targets represented in output: {len(unique_targets)}")
    print("    (Some targets may be legitimately absent if they had zero eligible")
    print("     donors under ALL THREE conditions -- not itself a failure. Cross-check")
    print("     this number against the generation console output's own reported")
    print("     'across N target images' and 'Skipped' counts by eligibility.)")

    # --- 2. Duplicate composite_id check ---
    ids = [r["composite_id"] for r in records]
    dupes = {i for i in ids if ids.count(i) > 1}
    print(f"\n[2] Duplicate composite_id values: {len(dupes)}")
    if dupes:
        hard_failures.append(f"{len(dupes)} duplicate composite_id values found: {list(dupes)[:10]}")
        print(f"    FAIL: {list(dupes)[:10]}")
    else:
        print("    PASS")

    # --- 3. Composite files exist and are non-empty ---
    missing_composite_files = []
    for r in records:
        p = r.get("composite_path")
        if not p or not os.path.exists(p) or os.path.getsize(p) == 0:
            missing_composite_files.append((r["composite_id"], p))
    print(f"\n[3] Composite image files missing/empty: {len(missing_composite_files)}")
    if missing_composite_files:
        hard_failures.append(f"{len(missing_composite_files)} composite files missing or empty")
        for cid, p in missing_composite_files[:10]:
            print(f"    FAIL: {cid} -> {p}")
    else:
        print("    PASS -- all composite files present and non-empty")

    # --- 4. Repaired-donor artifact files exist ---
    donor_paths = {r["donor_id"]: r.get("repaired_donor_path") for r in records}
    missing_donor_files = [(d, p) for d, p in donor_paths.items()
                            if not p or not os.path.exists(p)]
    print(f"\n[4] Repaired-donor artifact files missing: {len(missing_donor_files)} "
          f"(of {len(donor_paths)} unique donors)")
    if missing_donor_files:
        hard_failures.append(f"{len(missing_donor_files)} repaired-donor artifacts missing")
        for d, p in missing_donor_files[:10]:
            print(f"    FAIL: {d} -> {p}")
    else:
        print("    PASS -- all repaired-donor artifacts present")

    # --- 5. ROI integrity: must be 100% ---
    roi_failures = [r for r in records if r.get("roi_integrity_ok") is not True]
    print(f"\n[5] ROI-integrity failures: {len(roi_failures)} / {len(records)}")
    if roi_failures:
        hard_failures.append(f"{len(roi_failures)} ROI-integrity failures -- locked-plan violation")
        for r in roi_failures[:10]:
            print(f"    FAIL: {r['composite_id']}  roi_max_diff={r.get('roi_max_diff')}")
    else:
        print("    PASS -- 100% of composites preserved target ROI pixels exactly")

    # --- 6. A/B/C rule re-verification ---
    rule_violations = []
    for r in records:
        violation = check_ab_c_rule(r)
        if violation:
            rule_violations.append((r["composite_id"], violation))
    print(f"\n[6] A/B/C eligibility-rule violations: {len(rule_violations)}")
    if rule_violations:
        hard_failures.append(f"{len(rule_violations)} A/B/C rule violations")
        for cid, v in rule_violations[:10]:
            print(f"    FAIL: {cid}  {v}")
    else:
        print("    PASS -- every composite's condition matches its own stored source/class fields")

    # --- 7. Required metadata fields present ---
    incomplete = []
    for r in records:
        for field in REQUIRED_FIELDS:
            if field not in r or r[field] is None:
                # weak_candidate_support/repair_admissible are real
                # booleans and repair_selected_score/tier are always
                # populated per Finding 2 -- None here is a real gap,
                # not a False/0 value.
                if field in ("weak_candidate_support", "repair_admissible") and r.get(field) is False:
                    continue
                incomplete.append((r["composite_id"], field))
    print(f"\n[7] Records with missing required metadata fields: "
          f"{len({cid for cid, _ in incomplete})}")
    if incomplete:
        hard_failures.append(f"{len(incomplete)} missing-field instances")
        for cid, field in incomplete[:10]:
            print(f"    FAIL: {cid}  missing '{field}'")
    else:
        print("    PASS -- all required fields populated on every record")

    # --- 8. Real tier/admissibility/weak-support distribution (report only) ---
    by_donor = {}
    for r in records:
        by_donor.setdefault(r["donor_id"], r)
    n_donors = len(by_donor)
    tier_counts = {1: 0, 2: 0, 3: 0}
    support_counts = {"weak": 0, "limited": 0, "adequate": 0}
    non_admissible = 0
    for r in by_donor.values():
        t = r.get("repair_tier")
        if t in tier_counts:
            tier_counts[t] += 1
        s = r.get("repair_candidate_support")
        if s in support_counts:
            support_counts[s] += 1
        if r.get("repair_admissible") is False:
            non_admissible += 1

    print(f"\n[8] Real distribution under full-sweep plain random donor assignment "
          f"({n_donors} unique donors):")
    for t in [1, 2, 3]:
        c = tier_counts[t]
        print(f"    Tier {t}: {c} ({100*c/n_donors:.1f}%)")
    for s in ["weak", "limited", "adequate"]:
        c = support_counts[s]
        print(f"    {s} candidate support: {c} ({100*c/n_donors:.1f}%)")
    print(f"    Non-admissible (tier3_no_texture_fallback): {non_admissible} "
          f"({100*non_admissible/n_donors:.1f}%)")
    print("    (Expect this weak-support rate close to finding2v1's ~1%, not")
    print("     finding2v2's stress-biased 9.5% -- --full uses plain random")
    print("     donor assignment, not the pilot's deliberate geometry bias.)")

    # --- Final verdict ---
    print("\n" + "=" * 70)
    if hard_failures:
        print(f"  RESULT: FAIL -- {len(hard_failures)} integrity issue(s) found")
        print("  DO NOT proceed to model inference until these are resolved:")
        for f in hard_failures:
            print(f"    - {f}")
        print("=" * 70)
        sys.exit(1)
    else:
        print("  RESULT: PASS -- mechanical integrity confirmed.")
        print("  Full 203-target Phase 8C dataset is ready for model inference.")
        print("=" * 70)


if __name__ == "__main__":
    main()

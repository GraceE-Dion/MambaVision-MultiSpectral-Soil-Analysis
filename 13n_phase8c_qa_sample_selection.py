"""
13n_phase8c_qa_sample_selection.py
=====================================
Reads composite_metadata.json from a Phase 8C pilot run and prints the
three QA sample groups peer review asked for, per composite ("adversarial
QA sample instead of an easy random sample"):

  1. Highest-score / worst-quality repairs (top N by repair_selected_score)
  2. ALL Tier-3 repairs (both sub-methods), if any occurred
  3. A heterogeneous sample of Tier-1/Tier-2 repairs spread across
     acquisition sources/modalities

Also separately flags:
  - Any composite with repair_admissible == False (must not enter B/C
    without a human decision, per peer review)
  - Tier-2 composites at the tightest shrink fractions (0.15, 0.08) --
    inspect these FIRST for residual donor-laser halo, since aggressive
    padding reduction is the scenario peer review specifically flagged.

Does not touch model weights or either conda env -- pure metadata/
filesystem read, safe to run from mambayolo or mambavision.

Usage:
    python 13n_phase8c_qa_sample_selection.py results/phase8c/pilot_finding2v1
"""

import os
import sys
import json
import random


def main():
    if len(sys.argv) < 2:
        print("Usage: python 13n_phase8c_qa_sample_selection.py <pilot_output_dir>")
        sys.exit(1)

    pilot_dir = sys.argv[1]
    metadata_path = os.path.join(pilot_dir, "composite_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"ERROR: {metadata_path} not found")
        sys.exit(1)

    with open(metadata_path) as f:
        records = json.load(f)

    print("=" * 70)
    print(f"  Phase 8C Pilot QA Sample Selection -- {len(records)} composites")
    print("=" * 70)

    def panel_path(r):
        return os.path.join(pilot_dir, f"{r['composite_id']}_QA_panel.png")

    # --- Flag: non-admissible repairs (must not silently enter B/C) ---
    non_admissible = [r for r in records if r.get("repair_admissible") is False]
    print(f"\n[FLAG] Non-admissible repairs (tier3_no_texture_fallback): {len(non_admissible)}")
    if non_admissible:
        print("  These must be pulled for a human decision before any full sweep:")
        for r in non_admissible:
            print(f"    {r['composite_id']}  (donor={r['donor_id']}, target={r['target_id']})  "
                  f"-> {panel_path(r)}")
    else:
        print("  None in this pilot.")

    # --- Flag: tightest-shrink Tier-2 (inspect first for residual halo) ---
    tight_shrink = [r for r in records
                     if r.get("repair_tier") == 2 and r.get("repair_shrink_fraction_used") in (0.15, 0.08)]
    print(f"\n[FLAG] Tier-2 at tightest shrink (0.15/0.08) -- inspect first for residual "
          f"donor-laser halo: {len(tight_shrink)}")
    seen_donors = set()
    for r in tight_shrink:
        if r["donor_id"] in seen_donors:
            continue
        seen_donors.add(r["donor_id"])
        print(f"    {r['composite_id']}  (donor={r['donor_id']}, shrink={r['repair_shrink_fraction_used']})  "
              f"-> {panel_path(r)}")

    # --- Group 1: highest-score (worst-match) repairs, one per donor ---
    by_donor = {}
    for r in records:
        d = r["donor_id"]
        if d not in by_donor or (r.get("repair_selected_score") or -1) > (by_donor[d].get("repair_selected_score") or -1):
            by_donor[d] = r
    ranked = sorted(by_donor.values(), key=lambda r: r.get("repair_selected_score") or -1, reverse=True)
    top_n = ranked[:8]
    print(f"\n[GROUP 1] Highest-score (worst-match) repairs -- top {len(top_n)} unique donors:")
    for r in top_n:
        print(f"    {r['composite_id']}  score={r['repair_selected_score']:.2f}  "
              f"tier={r['repair_tier']}  donor={r['donor_id']} ({r['donor_source']})  "
              f"-> {panel_path(r)}")

    # --- Group 2: ALL Tier-3 repairs ---
    tier3 = [r for r in records if r.get("repair_tier") == 3]
    print(f"\n[GROUP 2] ALL Tier-3 repairs: {len(tier3)}")
    if tier3:
        for r in tier3:
            print(f"    {r['composite_id']}  method={r['repair_method']}  "
                  f"admissible={r['repair_admissible']}  -> {panel_path(r)}")
    else:
        print("  None occurred in this pilot -- Tier 1/2 resolved every donor.")

    # --- Group 3: heterogeneous Tier-1/Tier-2 sample across sources ---
    random.seed(0)
    non_tier3 = [r for r in records if r.get("repair_tier") in (1, 2)]
    by_source = {}
    for r in non_tier3:
        by_source.setdefault(r["donor_source"], []).append(r)
    print(f"\n[GROUP 3] Heterogeneous Tier-1/Tier-2 sample across {len(by_source)} donor sources "
          f"(1-2 per source):")
    for src in sorted(by_source):
        pool = by_source[src]
        picks = random.sample(pool, min(2, len(pool)))
        for r in picks:
            print(f"    {r['composite_id']}  tier={r['repair_tier']}  score={r['repair_selected_score']:.2f}  "
                  f"donor_source={src}  -> {panel_path(r)}")

    print("\n" + "=" * 70)
    print("  Open the *_QA_panel.png files listed above (any local viewer, or copy")
    print("  the pilot dir back through git if easier to inspect from the laptop).")
    print("  For each: confirm target ROI matches exactly, donor's own laser spot")
    print("  is NOT visible anywhere in the composite, no rectangular seam artifact,")
    print("  and (for the tight-shrink flags) no residual donor-laser halo.")
    print("=" * 70)


if __name__ == "__main__":
    main()

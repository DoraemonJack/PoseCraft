"""Aggregate PlaceIt3D benchmark metrics.

The official JSON schema and the reductions below are fixed by a published
docker run (five val placements) and by the global-accuracy identity in
nianticlabs/placeit3d issue #5:

* global constraint accuracy is 100 * satisfied / total over every constraint,
  including physical plausibility;
* spatial, rotational and physical subgroup scores are micro-averages;
* visibility is the unweighted mean of the ``is_visible`` and ``not_visible``
  rates (PoseCraft supplement; both rates are defined on the full test set);
* language adherence and complete placement are percentages of placements;
* a placement with no language constraint is not language-adherent;
* ``metrics_by_constraint_count`` groups by the number of language constraints
  and omits the empty group.
"""
from __future__ import annotations

CANONICAL_CONSTRAINTS = (
    "near",
    "adjacent",
    "on",
    "above",
    "facing",
    "not_visible",
    "between",
    "is_visible",
    "plausible",
    "below",
)

SPATIAL = {"near", "adjacent", "on", "above", "below", "between", "in"}
ROTATIONAL = {"facing", "parallel", "perpendicular"}
VISIBILITY = ("is_visible", "not_visible")


def _accuracy(satisfied: int, total: int):
    if total == 0:
        return None
    return satisfied / total


def aggregate(placements: list[dict]) -> dict:
    """Reduce per-placement constraint booleans to the official metrics dict.

    Each placement is ``{"language": [(name, ok), ...], "physical": bool}``.
    ``plausible`` entries inside ``language`` are ignored; physical plausibility
    is always taken from ``physical``.
    """
    if not placements:
        raise ValueError("No placements to aggregate.")
    counts = {name: [0, 0] for name in CANONICAL_CONSTRAINTS}
    buckets: dict[str, dict] = {}
    language_success = 0
    complete_success = 0
    for placement in placements:
        language = [(name, bool(ok)) for name, ok in placement["language"] if name != "plausible"]
        for name, ok in language:
            slot = counts.setdefault(name, [0, 0])
            slot[1] += 1
            slot[0] += int(ok)
        counts["plausible"][1] += 1
        counts["plausible"][0] += int(placement["physical"])
        count = len(language)
        if count == 0:
            continue
        bucket = buckets.setdefault(
            str(count),
            {"total_placements": 0, "at_least_one_correct": 0, "all_correct": 0},
        )
        bucket["total_placements"] += 1
        flags = [ok for _, ok in language]
        bucket["at_least_one_correct"] += int(any(flags))
        all_ok = all(flags)
        bucket["all_correct"] += int(all_ok)
        if all_ok:
            language_success += 1
            if placement["physical"]:
                complete_success += 1

    per_constraint = {}
    for name in CANONICAL_CONSTRAINTS:
        satisfied, total = counts[name]
        per_constraint[name] = {
            "accuracy": _accuracy(satisfied, total),
            "satisfied": satisfied,
            "total": total,
        }
    for name, (satisfied, total) in counts.items():
        if name not in per_constraint and total:
            per_constraint[name] = {
                "accuracy": _accuracy(satisfied, total),
                "satisfied": satisfied,
                "total": total,
            }

    def micro(names) -> float | None:
        satisfied = sum(counts[name][0] for name in names if name in counts)
        total = sum(counts[name][1] for name in names if name in counts)
        return _accuracy(satisfied, total)

    visible = per_constraint["is_visible"]["accuracy"]
    hidden = per_constraint["not_visible"]["accuracy"]
    defined = [value for value in (visible, hidden) if value is not None]
    # The official logger adds the two rates and crashes when either class is
    # absent (accuracy is null). Both classes occur on the released test set,
    # where the mean is well defined. A missing class uses the class that ran.
    visibility = None if not defined else float(sum(defined) / len(defined))
    total_satisfied = sum(slot[0] for slot in counts.values())
    total = sum(slot[1] for slot in counts.values())
    n = len(placements)
    ordered_buckets = {key: buckets[key] for key in sorted(buckets, key=int)}
    return {
        "language_adherence_success": 100.0 * language_success / n,
        "global_constraint_accuracy": 100.0 * total_satisfied / total,
        "complete_placement_success": 100.0 * complete_success / n,
        "subgroup_metrics": {
            "spatial": micro(SPATIAL),
            "rotational": micro(ROTATIONAL),
            "visibility": visibility,
            "physical": micro(("plausible",)),
        },
        "per_constraint_metrics": per_constraint,
        "metrics_by_constraint_count": ordered_buckets,
    }


def format_summary(metrics: dict) -> str:
    lines = [
        f"Language Adherence Success: {metrics['language_adherence_success']:.2f}",
        f"Global Constraint Accuracy: {metrics['global_constraint_accuracy']:.2f}",
        f"Complete Placement Success: {metrics['complete_placement_success']:.2f}",
        "Subgroup Metrics:",
    ]
    for name, value in metrics["subgroup_metrics"].items():
        rendered = "n/a" if value is None else f"{100.0 * value:.2f}"
        lines.append(f"\t{name}: {rendered}")
    return "\n".join(lines)

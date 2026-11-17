"""Open reimplementation of the PlaceIt3D rule-based benchmark.

The public repository ships the evaluator only as ``aabdelreheem/placeit3d-benchmark``.
This program reads the same ``predictions.json`` records (``entry_id``, ``tx``,
``ty``, ``tz``, ``rotation``), the released HDF5 annotations, and ScanNet scans,
then writes the same aggregate metric object.

Two defaults differ from a literal reading of the supplement, because that reading
disagrees with the released masks and with a docker smoke run. ``near`` uses 10%
of the room's horizontal diagonal (pass ``--near-fraction 0.01`` for the written
1%). Visibility shoots rays at asset samples instead of a 256x256 pyrender
image; a missing visibility class is reported as null instead of raising.

Example::

    python evaluate.py val --predictions predictions.json --h5 placeit3d_val.h5 ^
        --scans /path/to/scans --output results
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from constraints import (
    DEFAULT_NEAR_ROOM_FRACTION,
    is_above,
    is_adjacent,
    is_below,
    is_between,
    is_facing,
    is_inside,
    is_near,
    is_on,
    is_parallel,
    is_perpendicular,
    near_threshold,
)
from data import Entry, SceneGeometry, load_asset_mesh, load_entries, load_scene, pose_asset
from geometry import OBB
from metrics import aggregate, format_summary
from local_floor import floor_on, FloorFitUnavailable
from physical import physically_plausible
from visibility import asset_is_visible

LOGGER = logging.getLogger("compute_metrics_parallel")


def _language_ok(name: str, asset: OBB, anchors: list[OBB], threshold: float) -> bool:
    if name == "near":
        return is_near(asset, anchors[0], threshold)
    if name == "adjacent":
        return is_adjacent(asset, anchors[0])
    if name == "on":
        return is_on(asset, anchors[0])
    if name == "above":
        return is_above(asset, anchors[0])
    if name == "below":
        return is_below(asset, anchors[0])
    if name == "between":
        if len(anchors) < 2:
            raise ValueError("between requires two anchors")
        return is_between(asset, anchors[0], anchors[1])
    if name == "facing":
        return is_facing(asset, anchors[0])
    if name == "in":
        return is_inside(asset, anchors[0])
    if name == "parallel":
        return is_parallel(asset, anchors[0])
    if name == "perpendicular":
        return is_perpendicular(asset, anchors[0])
    raise KeyError(f"Unsupported relation {name}")


def _safe(name: str, function):
    try:
        return bool(function())
    except FloorFitUnavailable as exc:
        LOGGER.warning("on fit unavailable: %s", exc)
        return False
    except Exception as exc:
        LOGGER.warning("Error evaluating constraint %s: %s", name, exc)
        return False


def evaluate_predictions(h5_path, predictions, scans_root, assets_root=None, aggregation_root=None,
                         segments_root=None, object_csv_root=None, object_sizes_path=None,
                         alignment_json=None, near_fraction=DEFAULT_NEAR_ROOM_FRACTION,
                         visibility_samples=256, check_visibility=True, check_physical=True,
                         asset_meshes=None):
    if isinstance(predictions, (str, Path)):
        predictions = json.loads(Path(predictions).read_text(encoding="utf-8"))
    if alignment_json is not None and not isinstance(alignment_json, dict):
        alignment_json = json.loads(Path(alignment_json).read_text(encoding="utf-8"))
    entries = load_entries(h5_path, object_sizes_path)
    scenes: dict[str, SceneGeometry] = {}
    asset_templates = {}
    reduced = []
    details = []
    for row in predictions:
        entry: Entry = entries[int(row["entry_id"])]
        scene = scenes.get(entry.scene_id)
        if scene is None:
            scene = load_scene(
                entry.scene_id, scans_root, aggregation_root, segments_root,
                object_csv_root, alignment_json,
            )
            scenes[entry.scene_id] = scene
        center = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
        yaw = float(row["rotation"])
        asset_box = OBB(center=center, size=entry.asset_size, yaw=np.deg2rad(yaw))
        template = asset_templates.get(entry.asset_uuid)
        if template is None:
            override = None if asset_meshes is None else asset_meshes.get(entry.asset_uuid)
            template = override if override is not None else load_asset_mesh(
                assets_root, entry.asset_uuid, entry.asset_size,
            )
            asset_templates[entry.asset_uuid] = template
        posed = pose_asset(template, center, yaw)
        threshold = near_threshold(scene.vertices, near_fraction)
        language = []
        detail_language = []
        for relation in entry.relationships:
            name = relation["relation"]
            if name == "plausible":
                continue
            anchors = relation["anchors"]

            def run(name=name, anchors=anchors):
                boxes = [scene.obb(object_id) for object_id in anchors]
                if name in ("is_visible", "not_visible"):
                    if not check_visibility:
                        raise RuntimeError("visibility check disabled")
                    visible = asset_is_visible(
                        scene.triangles, scene.tri_min, scene.tri_max,
                        np.asarray(posed.triangles, dtype=np.float64),
                        scene.anchor_vertices.get(anchors[0]),
                        asset_box, samples=visibility_samples,
                    )
                    return visible if name == "is_visible" else not visible
                if name == 'on' and anchors:
                    return floor_on(asset_box, boxes[0], scene.anchor_vertices.get(int(anchors[0])))
                return _language_ok(name, asset_box, boxes, threshold)

            ok = _safe(name, run)
            language.append((name, ok))
            detail_language.append({"relation": name, "anchors": anchors, "satisfied": ok})
        physical = _safe("plausible", lambda: physically_plausible(
            scene.mesh, posed, asset_box, scene.tri_min, scene.tri_max,
        )) if check_physical else False
        reduced.append({"language": language, "physical": physical})
        details.append({
            "entry_id": int(row["entry_id"]),
            "scene_id": entry.scene_id,
            "asset_uuid": entry.asset_uuid,
            "prompt": entry.prompt,
            "physical": physical,
            "near_threshold_m": threshold,
            "language": detail_language,
        })
    return aggregate(reduced), details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PlaceIt3D rule-based evaluation")
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--scans", required=True, help="ScanNet scans directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", default=None, help="Directory of *_scaled_aligned.glb assets")
    parser.add_argument("--aggregation-root", default=None)
    parser.add_argument("--segments-root", default=None)
    parser.add_argument("--object-csv-root", default=None, help="VLA-3D per-scene object_result.csv files")
    parser.add_argument("--object-sizes", default=None)
    parser.add_argument("--alignment-json", default=None)
    parser.add_argument("--near-fraction", type=float, default=DEFAULT_NEAR_ROOM_FRACTION)
    parser.add_argument("--visibility-samples", type=int, default=256)
    parser.add_argument("--skip-visibility", action="store_true")
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
    metrics, details = evaluate_predictions(
        args.h5, args.predictions, args.scans, args.assets, args.aggregation_root,
        args.segments_root, args.object_csv_root, args.object_sizes, args.alignment_json,
        args.near_fraction, args.visibility_samples, not args.skip_visibility, not args.skip_physical,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.details:
        (output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(format_summary(metrics))
    print(metrics)
    print(output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the three epoch-8 ablations with the same settings as the full test."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import trimesh

import constraints
from constraints import near_threshold
import logging
from local_floor import floor_on, FloorFitUnavailable
from data import load_asset_mesh, load_entries, load_scene, pose_asset
from evaluate import _language_ok
from geometry import OBB
from metrics import aggregate, format_summary
from physical import physically_plausible
from visibility import asset_is_visible

ASSETS = Path("/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1/geometry/assets")
OUT = Path("/path/to/home/placeit3d_near10_run/results")
H5 = Path("/path/to/home/Downloads/placeit3d_test.h5")
SCANS = Path("/path/to/research300/refineplace_full3500_eval_20261104_v1/scans")
WORKERS = int(os.environ.get('WORKERS', '6'))
NEAR_FRACTION = float(os.environ.get("NEAR_FRACTION", "0.10"))
constraints.FACING_ANGLE_RAD = np.deg2rad(float(os.environ.get("FACING_DEG", "20")))
LOGGER = logging.getLogger("compute_metrics_parallel")
JOBS = (
    ("ablation_no_action", Path("/path/to/research300/refineplace_v20_epoch8_ablation_20261108/no_action_adjustment.json")),
    ("ablation_no_orientation", Path("/path/to/research300/refineplace_v20_epoch8_ablation_20261108/no_orientation_refinement.json")),
    ("ablation_no_both", Path("/path/to/research300/posecraft_adaptive_stopping_20261114_nmax4/no_both_initial_predictions_3500.json")),
)

ENTRIES = None
MESHES = None
NAME = ""


def load_predictions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    rows = sorted(rows, key=lambda row: int(row["entry_id"]))
    ids = [int(row["entry_id"]) for row in rows]
    if ids != list(range(len(ENTRIES))):
        raise ValueError(f"{path} entry ids n={len(ids)}, h5 has {len(ENTRIES)}")
    return rows


def _mesh(uuid: str, size: np.ndarray) -> trimesh.Trimesh:
    cached = MESHES.get(uuid)
    if cached is not None:
        return cached
    path = ASSETS / f"{uuid}.npz"
    if path.is_file():
        archive = np.load(path)
        mesh = trimesh.Trimesh(archive["vertices"], archive["faces"], process=False)
    else:
        mesh = load_asset_mesh(ASSETS, uuid, size)
    MESHES[uuid] = mesh
    return mesh


def _one(scene, entry, row) -> dict:
    center = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
    yaw = float(row["rotation"])
    asset_box = OBB(center=center, size=entry.asset_size, yaw=np.deg2rad(yaw))
    posed = pose_asset(_mesh(entry.asset_uuid, entry.asset_size), center, yaw)
    threshold = near_threshold(scene.vertices, NEAR_FRACTION)
    language = []
    for relation in entry.relationships:
        name = relation["relation"]
        if name == "plausible":
            continue
        anchors = relation["anchors"]
        try:
            if name in ("is_visible", "not_visible"):
                visible = asset_is_visible(
                    scene.triangles, scene.tri_min, scene.tri_max,
                    np.asarray(posed.triangles, dtype=np.float64),
                    scene.anchor_vertices.get(anchors[0]),
                    asset_box, samples=256,
                )
                ok = visible if name == "is_visible" else (not visible)
            elif name == "on" and anchors:
                try:
                    ok = floor_on(asset_box, scene.obb(anchors[0]), scene.anchor_vertices.get(int(anchors[0])))
                except FloorFitUnavailable as exc:
                    LOGGER.warning("on fit unavailable: %s", exc)
                    ok = False
            else:
                ok = _language_ok(name, asset_box, [scene.obb(object_id) for object_id in anchors], threshold)
        except Exception as exc:
            language.append((name, False, str(exc)))
            continue
        language.append((name, bool(ok), ""))
    try:
        physical = bool(physically_plausible(scene.mesh, posed, asset_box, scene.tri_min, scene.tri_max))
    except Exception:
        physical = False
    return {
        "entry_id": entry.index,
        "language": [(name, ok) for name, ok, _ in language],
        "errors": [f"{name}: {err}" for name, _, err in language if err],
        "physical": physical,
    }


def _scene_job(payload):
    scene_id, rows, scans = payload
    started = time.perf_counter()
    try:
        scene = load_scene(scene_id, scans)
        done = [_one(scene, ENTRIES[int(row["entry_id"])], row) for row in rows]
        return {"scene_id": scene_id, "rows": done, "seconds": time.perf_counter() - started, "error": ""}
    except Exception:
        return {
            "scene_id": scene_id,
            "rows": [],
            "seconds": time.perf_counter() - started,
            "error": traceback.format_exc(),
        }


def evaluate(name: str, predictions: Path) -> None:
    global NAME
    NAME = name
    print(f"{name} entries {len(ENTRIES)}", flush=True)
    rows = load_predictions(predictions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[ENTRIES[int(row["entry_id"])].scene_id].append(row)
    jobs = [(scene_id, scene_rows, str(SCANS)) for scene_id, scene_rows in grouped.items()]
    reduced = []
    failures = []
    started = time.perf_counter()
    with get_context("fork").Pool(WORKERS) as pool:
        for index, result in enumerate(pool.imap_unordered(_scene_job, jobs), start=1):
            if result["error"]:
                failures.append({"scene_id": result["scene_id"], "error": result["error"]})
                print(f"{name} scene {result['scene_id']} FAILED", flush=True)
            else:
                reduced.extend(result["rows"])
            if index % 10 == 0 or index == len(jobs):
                print(
                    f"{name} scenes {index}/{len(jobs)} placements {len(reduced)} "
                    f"elapsed {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    reduced.sort(key=lambda item: item["entry_id"])
    metrics = aggregate(reduced)
    errors = sum(len(item["errors"]) for item in reduced)
    (OUT / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / f"{name}_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(format_summary(metrics), flush=True)
    print(
        f"{name} constraint_errors {errors} scene_failures {len(failures)} "
        f"placements {len(reduced)}/{len(rows)}",
        flush=True,
    )


def main() -> None:
    global ENTRIES, MESHES
    MESHES = {}
    ENTRIES = {entry.index: entry for entry in load_entries(H5)}
    print(
        f"trial near={NEAR_FRACTION} facing_deg={os.environ.get('FACING_DEG', '20')}",
        flush=True,
    )
    evaluate(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()

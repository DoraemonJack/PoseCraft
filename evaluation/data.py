"""Load PlaceIt3D HDF5 entries, ScanNet geometry, and aligned assets.

Entry order follows ``PlacementDataset`` in the public repository: groups are
visited with ``h5py``'s stored order, and ``entry_id`` in ``predictions.json``
is that position. Anchor ids are ScanNet ``objectId`` values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import trimesh

from geometry import OBB, rotz


@dataclass
class Entry:
    index: int
    scene_id: str
    asset_uuid: str
    asset_size: np.ndarray
    relationships: list
    prompt: str


@dataclass
class SceneGeometry:
    scene_id: str
    vertices: np.ndarray
    triangles: np.ndarray
    tri_min: np.ndarray
    tri_max: np.ndarray
    boxes: dict
    anchor_vertices: dict
    mesh: trimesh.Trimesh
    labels: dict = field(default_factory=dict)

    def obb(self, object_id: int) -> OBB:
        try:
            return self.boxes[int(object_id)]
        except KeyError as exc:
            raise KeyError(f"Anchor {object_id} is missing in {self.scene_id}") from exc


def load_entries(h5_path: str | Path, object_sizes_path: str | Path | None = None) -> list[Entry]:
    sizes = {}
    if object_sizes_path:
        sizes = json.loads(Path(object_sizes_path).read_text(encoding="utf-8"))
    entries = []
    with h5py.File(h5_path, "r") as handle:
        relation_types = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["metadata/relation_types"][()]
        ]
        for asset_name, asset_group in handle.items():
            if asset_name == "metadata" or not isinstance(asset_group, h5py.Group):
                continue
            parts = asset_name.split("_")
            scene_id = "_".join(parts[:2]) if len(parts) >= 3 else asset_name
            asset_uuid = "_".join(parts[2:]) if len(parts) >= 3 else asset_name
            if "asset_size" in asset_group:
                asset_size = np.asarray(asset_group["asset_size"][:], dtype=np.float64)
            else:
                asset_size = _size_from_catalog(sizes, asset_uuid)
            for entry_name, entry_group in asset_group.items():
                if not isinstance(entry_group, h5py.Group) or "relationships" not in entry_group:
                    continue
                prompt = entry_group["prompt"][()]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
                relationships = []
                for row in entry_group["relationships"][:]:
                    name = relation_types[int(row[0])]
                    anchors = [int(value) for value in row[1:] if int(value) != -1]
                    relationships.append({"relation": name, "anchors": anchors})
                entries.append(Entry(
                    index=len(entries),
                    scene_id=scene_id,
                    asset_uuid=asset_uuid,
                    asset_size=asset_size,
                    relationships=relationships,
                    prompt=str(prompt),
                ))
    if not entries:
        raise ValueError(f"No entries in {h5_path}")
    return entries


def _size_from_catalog(sizes: dict, asset_uuid: str) -> np.ndarray:
    for key, value in sizes.items():
        if asset_uuid in key:
            return np.asarray(value["size"], dtype=np.float64)
    raise KeyError(f"No asset size for {asset_uuid}")


def _find(roots: list[Path], scene_id: str, filename: str) -> Path | None:
    for root in roots:
        if root is None:
            continue
        for candidate in (root / scene_id / filename, root / filename):
            if candidate.is_file():
                return candidate
    return None


def alignment_matrix(scene_id: str, scans_root: Path, alignment_json: dict | None) -> np.ndarray:
    if alignment_json and scene_id in alignment_json:
        return np.asarray(alignment_json[scene_id], dtype=np.float64).reshape(4, 4)
    txt = _find([scans_root], scene_id, f"{scene_id}.txt")
    matrix = np.eye(4)
    if txt is None:
        return matrix
    for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if "axisAlignment" not in line:
            continue
        values = [float(item) for item in line.split("axisAlignment = ", 1)[1].split()]
        return np.asarray(values, dtype=np.float64).reshape(4, 4)
    return matrix


def load_scene(scene_id: str, scans_root: str | Path, aggregation_root: str | Path | None = None,
               segments_root: str | Path | None = None, object_csv_root: str | Path | None = None,
               alignment_json: dict | None = None) -> SceneGeometry:
    scans = Path(scans_root)
    mesh_path = _find([scans], scene_id, f"{scene_id}_vh_clean_2.ply")
    if mesh_path is None:
        raise FileNotFoundError(f"Missing mesh for {scene_id} under {scans}")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    matrix = alignment_matrix(scene_id, scans, alignment_json)
    homo = np.ones((len(vertices), 4))
    homo[:, :3] = vertices
    aligned = (homo @ matrix.T)[:, :3]
    mesh = trimesh.Trimesh(vertices=aligned, faces=np.asarray(mesh.faces), process=False)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    boxes, anchors, labels = _anchors(
        scene_id, aligned, scans, Path(aggregation_root) if aggregation_root else None,
        Path(segments_root) if segments_root else None,
        Path(object_csv_root) if object_csv_root else None,
    )
    return SceneGeometry(
        scene_id=scene_id,
        vertices=aligned,
        triangles=triangles,
        tri_min=triangles.min(1),
        tri_max=triangles.max(1),
        boxes=boxes,
        anchor_vertices=anchors,
        mesh=mesh,
        labels=labels,
    )


def _anchors(scene_id, vertices, scans, aggregation_root, segments_root, csv_root):
    agg_roots = [path for path in (scans, aggregation_root) if path is not None]
    seg_roots = [path for path in (scans, segments_root) if path is not None]
    agg_path = _find(agg_roots, scene_id, f"{scene_id}.aggregation.json")
    seg_path = _find(seg_roots, scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    boxes, anchors, labels = {}, {}, {}
    if agg_path and seg_path:
        aggregation = json.loads(agg_path.read_text(encoding="utf-8"))
        segments = np.asarray(json.loads(seg_path.read_text(encoding="utf-8"))["segIndices"], dtype=np.int64)
        max_segment = int(max(max(group["segments"]) for group in aggregation["segGroups"]))
        seg_to_object = np.full(max(max_segment, int(segments.max())) + 1, -1, dtype=np.int32)
        for group in aggregation["segGroups"]:
            seg_to_object[np.asarray(group["segments"], dtype=np.int64)] = int(group["objectId"])
            labels[int(group["objectId"])] = group.get("label", "")
        object_ids = seg_to_object[segments]
        for object_id, label in labels.items():
            points = vertices[object_ids == object_id]
            if len(points) == 0:
                continue
            low, high = points.min(0), points.max(0)
            boxes[object_id] = OBB(center=(low + high) / 2.0, size=high - low, yaw=0.0)
            anchors[object_id] = points
    csv_path = None
    if csv_root is not None:
        csv_path = _find([csv_root, scans], scene_id, f"{scene_id}_object_result.csv")
    if csv_path is not None:
        _overlay_csv_boxes(csv_path, boxes, labels)
    if not boxes:
        raise FileNotFoundError(f"No anchor boxes for {scene_id}. Provide aggregation+segments or a VLA object CSV.")
    return boxes, anchors, labels


def _overlay_csv_boxes(path: Path, boxes: dict, labels: dict):
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            object_id = int(float(row["object_id"]))
            center = np.array([float(row["object_bbox_cx"]), float(row["object_bbox_cy"]), float(row["object_bbox_cz"])])
            size = np.array([
                float(row["object_bbox_xlength"]),
                float(row["object_bbox_ylength"]),
                float(row["object_bbox_zlength"]),
            ])
            boxes[object_id] = OBB(center=center, size=size, yaw=float(row["object_bbox_heading"]))
            labels[object_id] = row.get("raw_label") or row.get("nyu_label") or labels.get(object_id, "")


def load_asset_mesh(assets_root: str | Path | None, asset_uuid: str, asset_size: np.ndarray) -> trimesh.Trimesh:
    if assets_root is not None:
        root = Path(assets_root)
        matches = [path for path in root.rglob(f"*{asset_uuid}*.glb") if path.is_file()]
        matches.sort(key=lambda path: (0 if "scaled_aligned" in path.name else 1, len(path.name)))
        if matches:
            loaded = trimesh.load(matches[0], process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.to_geometry()
            if isinstance(loaded, (list, tuple)):
                loaded = trimesh.util.concatenate(loaded)
            return loaded
    return trimesh.creation.box(extents=np.asarray(asset_size, dtype=np.float64))


def pose_asset(mesh: trimesh.Trimesh, center: np.ndarray, yaw_degrees: float) -> trimesh.Trimesh:
    """Rotate about the origin, then move the mesh so its bounds center is ``center``.

    Released ``*_scaled_aligned.glb`` files are already centered. Meshes whose
    origin sits on the bottom are recentered first, because predictions give
    the asset center.
    """
    posed = mesh.copy()
    origin = posed.bounds.mean(0)
    if np.linalg.norm(origin) > 1e-8:
        posed.apply_translation(-origin)
    transform = np.eye(4)
    transform[:3, :3] = rotz(np.deg2rad(yaw_degrees))
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    posed.apply_transform(transform)
    return posed

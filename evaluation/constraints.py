"""Language-constraint predicates from PlaceIt3D supplement section 9.1.

``near`` stays proportional to the room, as the supplement states. The written
factor is 1% of the room size. On the released validation masks the farthest
accepted surface gap is about 16% of the horizontal room diagonal, and an
official docker run accepted a near placement at about 9% of that diagonal in
``scene0011_00`` (1% of the diagonal is 0.105 m and would reject it). The
default factor is 0.10 of the room diagonal, and the facing angle is 20 degrees. Pass ``near_fraction=0.01`` to use the
literal supplement number.
"""
from __future__ import annotations

import numpy as np

from geometry import OBB, box_gap, forward_angle, front_vector, iom_1d, lateral_iom, point_in_obb, xy_iom, xy_polygon, z_interval
from geometry import convex_hull, intersection_over_minimum

DEFAULT_NEAR_ROOM_FRACTION = 0.10
ADJACENT_TOLERANCE_M = 0.03
VERTICAL_IOM_MIN = 0.5
VERTICAL_SEPARATION_M = 0.01
ON_TOLERANCE_M = 0.01
BETWEEN_IOM_MIN = 0.5
BETWEEN_OVERLAP_MAX = 0.3
BETWEEN_DISTANCE_M = 1.5
FACING_DISTANCE_M = 2.0
FACING_ANGLE_RAD = np.deg2rad(20.0)
FACING_LATERAL_IOM_MIN = 0.5
HEADING_TOLERANCE_RAD = np.deg2rad(15.0)


def room_horizontal_diagonal(vertices: np.ndarray) -> float:
    span = np.asarray(vertices, dtype=np.float64).max(0) - np.asarray(vertices).min(0)
    return float(np.hypot(span[0], span[1]))


def near_threshold(vertices: np.ndarray, fraction: float = DEFAULT_NEAR_ROOM_FRACTION) -> float:
    return float(fraction) * room_horizontal_diagonal(vertices)


def is_near(asset: OBB, anchor: OBB, threshold: float) -> bool:
    return box_gap(asset, anchor) <= threshold


def is_adjacent(asset: OBB, anchor: OBB, tolerance: float = ADJACENT_TOLERANCE_M) -> bool:
    return box_gap(asset, anchor) <= tolerance


def is_on(asset: OBB, anchor: OBB) -> bool:
    asset_bottom, _ = z_interval(asset)
    _, anchor_top = z_interval(anchor)
    return xy_iom(asset, anchor) >= VERTICAL_IOM_MIN and abs(asset_bottom - anchor_top) <= ON_TOLERANCE_M


def is_above(asset: OBB, anchor: OBB) -> bool:
    asset_bottom, _ = z_interval(asset)
    _, anchor_top = z_interval(anchor)
    return xy_iom(asset, anchor) >= VERTICAL_IOM_MIN and asset_bottom >= anchor_top + VERTICAL_SEPARATION_M


def is_below(asset: OBB, anchor: OBB) -> bool:
    _, asset_top = z_interval(asset)
    anchor_bottom, _ = z_interval(anchor)
    return xy_iom(asset, anchor) >= VERTICAL_IOM_MIN and asset_top <= anchor_bottom - VERTICAL_SEPARATION_M


def is_between(asset: OBB, first: OBB, second: OBB) -> bool:
    """Asset lies between two anchors in the xy and z projections.

    The supplement requires IoM of at least 0.5 on both projection planes, IoM
    of at most 0.3 against either anchor, and a 1.5 m distance filter.
    """
    if box_gap(asset, first) > BETWEEN_DISTANCE_M or box_gap(asset, second) > BETWEEN_DISTANCE_M:
        return False
    start = first.center[:2]
    end = second.center[:2]
    offset = end - start
    denom = float(np.dot(offset, offset))
    if denom <= 1e-10:
        return False
    t = float(np.dot(asset.center[:2] - start, offset) / denom)
    if not 0.0 <= t <= 1.0:
        return False
    if xy_iom(asset, first) > BETWEEN_OVERLAP_MAX or xy_iom(asset, second) > BETWEEN_OVERLAP_MAX:
        return False
    region = convex_hull(np.vstack([xy_polygon(first), xy_polygon(second)]))
    if intersection_over_minimum(xy_polygon(asset), region) < BETWEEN_IOM_MIN:
        return False
    z0 = min(z_interval(first)[0], z_interval(second)[0])
    z1 = max(z_interval(first)[1], z_interval(second)[1])
    return iom_1d(*z_interval(asset), z0, z1) >= BETWEEN_IOM_MIN


def is_facing(asset: OBB, anchor: OBB) -> bool:
    return (
        box_gap(asset, anchor) <= FACING_DISTANCE_M
        and forward_angle(asset, anchor) <= FACING_ANGLE_RAD
        and lateral_iom(asset, anchor) >= FACING_LATERAL_IOM_MIN
    )


def is_inside(asset: OBB, anchor: OBB) -> bool:
    return bool(point_in_obb(asset.center.reshape(1, 3), anchor)[0]) and xy_iom(asset, anchor) >= VERTICAL_IOM_MIN


def _heading_delta(asset: OBB, anchor: OBB) -> float:
    delta = abs((asset.yaw - anchor.yaw + np.pi) % (2.0 * np.pi) - np.pi)
    return float(min(delta, np.pi - delta))


def is_parallel(asset: OBB, anchor: OBB) -> bool:
    return _heading_delta(asset, anchor) <= HEADING_TOLERANCE_RAD


def is_perpendicular(asset: OBB, anchor: OBB) -> bool:
    return abs(_heading_delta(asset, anchor) - np.pi / 2.0) <= HEADING_TOLERANCE_RAD


def asset_front_yaw(yaw_radians: float) -> np.ndarray:
    return front_vector(yaw_radians)

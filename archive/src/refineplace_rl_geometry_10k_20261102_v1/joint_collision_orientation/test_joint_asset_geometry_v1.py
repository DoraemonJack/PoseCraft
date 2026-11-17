"""Analytic fixtures for geometry semantics, transforms and adapter isolation."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'rl_placement_v1'))
from asset_mesh_geometry_v6 import AssetMeshGeometry as OriginalAssetMeshGeometry
from joint_asset_geometry_v1 import (AssetMeshGeometry, DIRECTIONAL_FEATURE_DIM,
                                    DIRECTIONAL_FEATURE_NAMES,
                                    _closest_points_on_triangles,
                                    install_on_adapter_module)


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    v = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1])
                  for x in (lo[0], hi[0])], float)
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
                  [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return v, f


def join(meshes):
    vertices, faces, n = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f+n)
        n += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def floor(z=0):
    return (np.array([[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]], float),
            np.array([[0, 1, 2], [0, 2, 3]]))


def wall(x):
    return (np.array([[x, -.7, -.2], [x, .7, -.2], [x, .7, .7], [x, -.7, .7]]),
            np.array([[0, 1, 2], [0, 2, 3]]))


class JointAssetGeometryTests(unittest.TestCase):
    asset = box([-.2, -.1, 0], [.2, .1, .3])

    def assertOriginalIdentical(self, original, augmented):
        self.assertTrue(set(original).issubset(augmented))
        for key, expected in original.items():
            actual = augmented[key]
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(expected, actual, err_msg=key)
            else:
                self.assertEqual(expected, actual, key)

    def test_original_support_and_crossings_are_bitwise_preserved(self):
        scene = join([floor(), wall(.15)])
        old = OriginalAssetMeshGeometry(*scene, *self.asset)
        new = AssetMeshGeometry(*scene, *self.asset)
        for center, yaw in (([0, 0, 0], 0.), ([.1, -.02, .019], .4),
                            ([0, 0, .021], math.pi/2), ([0, 0, .2], -.2)):
            self.assertOriginalIdentical(old.query(center, yaw), new.query(center, yaw))
        np.testing.assert_array_equal(old.support_samples, new.support_samples)

    def test_contact_floor_and_near_contact_floor_are_not_collision_cost(self):
        for z in (0., .019, -.019):
            geom = AssetMeshGeometry(*floor(z), *self.asset)
            result = geom.query([0, 0, 0], 0.)
            self.assertEqual(result['support_contact_fraction'], 1.)
            self.assertEqual(result['collision_cost'], 0.)
            self.assertFalse(result['local_scene_evidence_mask'])

    def test_open_scene_wall_has_graded_tolerance_adjusted_asset_intrusion(self):
        costs = []
        for intrusion in (.005, .015, .04, .055, .08):
            geom = AssetMeshGeometry(*wall(.2-intrusion), *self.asset)
            result = geom.query([0, 0, 0], 0.)
            costs.append(result['collision_cost'])
            self.assertTrue(result['collision_cost_known'])
            self.assertGreater(result['collision_noncontact_pairs'], 0)
            self.assertIsNone(result['penetration_depth_m'])
            self.assertFalse(result['collision_proxy_is_physical_penetration_depth'])
            self.assertFalse(result['collision_proxy_certifies_clearance'])
        self.assertEqual(costs[0], 0.)
        self.assertEqual(costs[1], 0.)
        self.assertTrue(all(a <= b for a, b in zip(costs, costs[1:])), costs)
        self.assertGreater(costs[2], 0.)
        self.assertLess(costs[2], costs[3])
        self.assertEqual(costs[-1], 1.)

    def test_cost_is_continuous_near_tolerance_not_binary_crossing(self):
        geom = AssetMeshGeometry(*wall(.15), *self.asset)
        values = [geom.joint_signals([x, 0, 0], 0.)['collision_cost']
                  for x in np.linspace(-.02, .01, 31)]
        self.assertGreater(len(set(np.round(values, 5))), 15)
        self.assertLess(np.max(np.abs(np.diff(values))), .05)

    def test_real_yaw_rotates_elongated_mesh_and_changes_obstacle_cost(self):
        asset = box([-.25, -.08, 0], [.25, .08, .3])
        geom = AssetMeshGeometry(*wall(.15), *asset)
        before = geom.query([0, 0, 0], 0.)
        after = geom.query([0, 0, 0], math.pi/2)
        self.assertGreater(before['collision_cost'], .5)
        self.assertEqual(after['collision_cost'], 0.)
        self.assertGreater(before['collision_noncontact_pairs'], 0)
        self.assertEqual(after['collision_noncontact_pairs'], 0)
        self.assertAlmostEqual(after['directional_collision_vector'][7], 1.)
        self.assertAlmostEqual(after['directional_collision_vector'][8], 0.)

    def test_open_asset_never_invents_signed_penetration(self):
        vertices, faces = self.asset
        geom = AssetMeshGeometry(*wall(.15), vertices, faces[:-1])
        result = geom.query([0, 0, 0], 0.)
        self.assertFalse(result['asset_closed_edge_topology'])
        self.assertFalse(result['closed_asset_interior_proxy_mask'])
        self.assertFalse(result['collision_cost_known'])
        self.assertEqual(result['collision_cost'], 0.)
        self.assertGreater(result['collision_noncontact_pairs'], 0)
        self.assertTrue(result['local_scene_evidence_mask'])
        self.assertGreater(max(result['directional_collision_vector'][3:7]), 0)
        self.assertIsNone(result['penetration_depth_m'])

    def test_missing_scene_is_unknown(self):
        geom = AssetMeshGeometry(np.zeros((0, 3)), np.zeros((0, 3), int), *self.asset)
        result = geom.query([0, 0, 0], 0.)
        self.assertEqual(result['mesh_clearance'], 'unknown')
        self.assertFalse(result['collision_cost_known'])
        self.assertFalse(result['local_scene_evidence_mask'])
        np.testing.assert_array_equal(result['directional_collision_vector'][3:7], np.zeros(4))
        self.assertFalse(result['observed_free_space'])

    def test_hole_between_feet_is_not_replaced_by_bounding_box(self):
        legs = [box([x-.04, y-.04, 0], [x+.04, y+.04, .4])
                for x in (-.22, .22) for y in (-.22, .22)]
        chair = join(legs+[box([-.3, -.3, .4], [.3, .3, .48])])
        obstacle = box([-.04, -.04, .12], [.04, .04, .22])
        actual = AssetMeshGeometry(*obstacle, *chair).query([0, 0, 0], 0.)
        solid = AssetMeshGeometry(*obstacle, *box([-.3, -.3, 0], [.3, .3, .48])).query([0, 0, 0], 0.)
        self.assertEqual(actual['collision_cost'], 0.)
        self.assertEqual(actual['mesh_surface_intersections'], 0)
        self.assertGreater(solid['collision_cost'], .8)

    def test_joint_rigid_scene_pose_transform_preserves_object_frame_signals(self):
        scene = wall(.16)
        a = AssetMeshGeometry(*scene, *self.asset).joint_signals([0, 0, 0], 0.)
        angle = .7
        rot = np.array([[math.cos(angle), -math.sin(angle), 0.],
                        [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        offset = np.array([1.3, -.7, .8])
        transformed = scene[0] @ rot.T+offset
        b = AssetMeshGeometry(transformed, scene[1], *self.asset).joint_signals(offset, angle)
        np.testing.assert_allclose(a['directional_collision_vector'][:7],
                                   b['directional_collision_vector'][:7], atol=1e-6)
        self.assertAlmostEqual(a['collision_cost'], b['collision_cost'], places=6)

    def test_winding_reversal_does_not_assign_scene_solidity(self):
        scene = wall(.15)
        a = AssetMeshGeometry(*scene, *self.asset).query([0, 0, 0], 0.)
        b = AssetMeshGeometry(scene[0], scene[1][:, ::-1],
                              self.asset[0], self.asset[1][:, ::-1]).query([0, 0, 0], 0.)
        np.testing.assert_allclose(a['directional_collision_vector'], b['directional_collision_vector'])
        self.assertFalse(b['watertight_scene_required_for_solid_containment'])

    def test_feature_schema_masks_and_direction_are_explicit(self):
        result = AssetMeshGeometry(*wall(.25), *self.asset).query([0, 0, 0], .0)
        vector = result['directional_collision_vector']
        self.assertEqual(vector.shape, (10,))
        self.assertEqual(DIRECTIONAL_FEATURE_DIM, 10)
        self.assertEqual(len(DIRECTIONAL_FEATURE_NAMES), 10)
        self.assertTrue(np.isfinite(vector).all())
        self.assertGreater(vector[3], vector[4])
        self.assertEqual(vector[0], result['collision_cost'])
        self.assertEqual(vector[1], float(result['local_scene_evidence_mask']))
        self.assertEqual(vector[2], float(result['closed_asset_interior_proxy_mask']))
        self.assertIn('no_semantic_front', result['directional_feature_frame'])

    def test_sample_cap_is_reported_and_does_not_change_exact_crossings(self):
        scene = join([wall(.15), wall(.14), wall(.13)])
        original = OriginalAssetMeshGeometry(*scene, *self.asset).query([0, 0, 0], 0.)
        augmented = AssetMeshGeometry(*scene, *self.asset, max_local_scene_triangles=1).query([0, 0, 0], 0.)
        self.assertOriginalIdentical(original, augmented)
        self.assertFalse(augmented['collision_proxy_sampling_complete'])
        self.assertEqual(augmented['collision_proxy_local_triangle_count'], 6)
        self.assertLessEqual(augmented['collision_proxy_scene_sample_count'], 8)

    def test_capped_zero_is_unknown_but_positive_intrusion_remains_evidence(self):
        clear_scene = join([wall(.25), wall(.26)])
        zero = AssetMeshGeometry(*clear_scene, *self.asset,
                                 max_local_scene_triangles=1).query([0, 0, 0], 0.)
        self.assertEqual(zero['collision_cost'], 0.)
        self.assertFalse(zero['collision_cost_known'])
        self.assertIn('triangle_cap', zero['collision_cost_unknown_reason'])
        collision_scene = join([wall(.15), wall(.14)])
        positive = AssetMeshGeometry(*collision_scene, *self.asset,
                                     max_local_scene_triangles=1).query([0, 0, 0], 0.)
        self.assertGreater(positive['collision_cost'], 0.)
        self.assertTrue(positive['collision_cost_known'])
        self.assertTrue(positive['collision_proxy_is_lower_bound_from_selected_samples'])

    def test_unresolved_thin_asset_is_unknown_and_preparation_cache_is_explicit(self):
        thin = box([-.2, -.0001, 0], [.2, .0001, .3])
        geom = AssetMeshGeometry(*wall(.15), *thin)
        result = geom.query([0, 0, 0], 0.)
        self.assertTrue(result['asset_closed_edge_topology'])
        self.assertFalse(result['closed_asset_interior_proxy_mask'])
        self.assertFalse(result['collision_cost_known'])
        self.assertGreater(result['collision_proxy_asset_preparation_seconds'], 0.)
        repeat = AssetMeshGeometry(*wall(.15), *thin).query([0, 0, 0], 0.)
        self.assertTrue(repeat['collision_proxy_asset_preparation_cache_hit'])
        self.assertFalse(geom.joint_asset['depth'].flags.writeable)

    def test_closed_body_with_open_attached_sheet_has_positive_only_partial_evidence(self):
        vertices, faces = self.asset
        attached_sheet = (np.array([vertices[0], vertices[1], [-.2, -.4, .15]]),
                          np.array([[0, 1, 2]]))
        asset = join([self.asset, attached_sheet])
        scene = wall(.15)
        geom = AssetMeshGeometry(*scene, *asset)
        result = geom.query([0, 0, 0], 0.)
        self.assertFalse(result['asset_closed_edge_topology'])
        self.assertTrue(result['closed_asset_interior_proxy_mask'])
        self.assertTrue(result['collision_proxy_partial_asset'])
        self.assertFalse(result['collision_proxy_full_asset'])
        self.assertEqual(result['collision_proxy_closed_component_count'], 1)
        self.assertGreater(result['collision_cost'], 0.)
        self.assertTrue(result['collision_cost_known'])
        self.assertGreater(result['collision_proxy_covered_surface_area_fraction'], 0.)
        self.assertLess(result['collision_proxy_covered_surface_area_fraction'], 1.)
        self.assertFalse(result['collision_proxy_holes_filled'])
        self.assertOriginalIdentical(OriginalAssetMeshGeometry(*scene, *asset).query([0, 0, 0], 0.), result)
        clear = AssetMeshGeometry(*wall(.25), *asset).query([0, 0, 0], 0.)
        self.assertEqual(clear['collision_cost'], 0.)
        self.assertFalse(clear['collision_cost_known'])
        self.assertEqual(clear['collision_cost_unknown_reason'], 'partial_asset_without_positive_intrusion')

    def test_nested_closed_shell_void_is_preserved_when_recovering_components(self):
        outer = box([-.3, -.3, 0], [.3, .3, .6])
        inner = box([-.12, -.12, .18], [.12, .12, .42])
        asset = join([outer, inner])
        obstacle_in_void = box([-.02, -.02, .27], [.02, .02, .33])
        result = AssetMeshGeometry(*obstacle_in_void, *asset).query([0, 0, 0], 0.)
        self.assertTrue(result['collision_proxy_full_asset'])
        self.assertEqual(result['collision_proxy_closed_component_count'], 2)
        self.assertEqual(result['collision_cost'], 0.)
        self.assertEqual(result['collision_noncontact_pairs'], 0)
        # Add an existing open piece: neither recovery nor partial masking may
        # turn the inner closed shell into a filled solid.
        open_piece = (np.array([[.3, -.3, 0.], [.3, .3, 0.], [.4, 0., .2]]),
                      np.array([[0, 1, 2]]))
        partial = AssetMeshGeometry(*obstacle_in_void, *join([asset, open_piece])).query([0, 0, 0], 0.)
        self.assertTrue(partial['collision_proxy_partial_asset'])
        self.assertEqual(partial['collision_cost'], 0.)
        self.assertFalse(partial['collision_cost_known'])

    def test_exact_duplicate_faces_are_removed_only_from_proxy(self):
        vertices, faces = self.asset
        duplicated = (vertices, np.concatenate([faces, faces[:2, ::-1]]))
        scene = wall(.15)
        old = OriginalAssetMeshGeometry(*scene, *duplicated).query([0, 0, 0], 0.)
        new = AssetMeshGeometry(*scene, *duplicated).query([0, 0, 0], 0.)
        self.assertOriginalIdentical(old, new)
        self.assertFalse(new['asset_closed_edge_topology'])
        self.assertTrue(new['collision_proxy_full_asset'])
        self.assertFalse(new['collision_proxy_partial_asset'])
        self.assertEqual(new['collision_proxy_duplicate_faces_removed'], 2)
        self.assertEqual(new['collision_proxy_covered_surface_area_fraction'], 1.)
        self.assertGreater(new['collision_cost'], 0.)

    def test_zero_volume_closed_edge_mesh_is_unresolved_not_a_crash(self):
        vertices = np.array([[0., -.1, 0.], [0., .1, 0.], [0., 0., .3]])
        faces = np.array([[0, 1, 2], [2, 1, 0]])
        result = AssetMeshGeometry(*wall(.05), vertices, faces).query([0, 0, 0], 0.)
        self.assertFalse(result['closed_asset_interior_proxy_mask'])
        self.assertFalse(result['collision_cost_known'])

    def test_triangle_closest_point_is_on_triangle_not_infinite_plane(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        np.testing.assert_allclose(_closest_points_on_triangles(triangle, [.2, .3, 2.]), [[.2, .3, 0.]])
        np.testing.assert_allclose(_closest_points_on_triangles(triangle, [2., 2., 1.]), [[.5, .5, 0.]])
        np.testing.assert_allclose(_closest_points_on_triangles(triangle, [-1., -1., 0.]), [[0., 0., 0.]])

    def test_install_is_explicit_idempotent_and_base_class_is_not_modified(self):
        adapter = SimpleNamespace(AssetMeshGeometry=OriginalAssetMeshGeometry)
        self.assertEqual(install_on_adapter_module(adapter), 10)
        self.assertEqual(install_on_adapter_module(adapter), 10)
        self.assertIs(adapter.AssetMeshGeometry, AssetMeshGeometry)
        self.assertFalse(hasattr(OriginalAssetMeshGeometry, 'joint_signals'))
        with self.assertRaises(ValueError):
            install_on_adapter_module(SimpleNamespace(AssetMeshGeometry=object))

    def test_actual_adapter_keeps_26_original_features(self):
        import mesh_scene_adapter_v6 as adapter_module
        from unittest.mock import patch
        scene = join([floor(), wall(.15)])
        library = SimpleNamespace(geometry_config={'collision_backend': 'sat'},
                                  asset=lambda uuid, dims: self.asset)
        dims = np.ptp(self.asset[0], axis=0)
        original = adapter_module.MeshSceneAdapterV6(*scene, library)
        original.set_asset('fixture', dims)
        expected = original.features([0, 0, 0], dims, 0.)
        with patch.object(adapter_module, 'AssetMeshGeometry', AssetMeshGeometry):
            augmented = adapter_module.MeshSceneAdapterV6(*scene, library)
            augmented.set_asset('fixture', dims)
            actual = augmented.features([0, 0, 0], dims, 0.)
        self.assertEqual(actual[0].shape, (26,))
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(actual[1]['directional_collision_vector'].shape, (10,))
        self.assertEqual(actual[1]['penetration_cost_alias'], 'binary_noncontact_triangle_crossing_not_depth')

    @unittest.skipUnless(importlib.util.find_spec('fcl'), 'optional python-fcl not installed')
    def test_fcl_original_result_preserved(self):
        scene = join([floor(), wall(.15)])
        old = OriginalAssetMeshGeometry(*scene, *self.asset, collision_backend='fcl')
        new = AssetMeshGeometry(*scene, *self.asset, collision_backend='fcl')
        self.assertOriginalIdentical(old.query([0, 0, 0], .3), new.query([0, 0, 0], .3))


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_geometry_coverage import component_topology, locked_prefix, recorded_poses
from test_joint_asset_geometry_v1 import box, join


class GeometryCoverageAuditTests(unittest.TestCase):
    def test_closed_component_recovery_is_diagnostic_only(self):
        closed = box([-.2, -.1, 0], [.2, .1, .3])
        sheet = (np.array([[1., 0., 0.], [2., 0., 0.], [1., 1., 0.]]),
                 np.array([[0, 1, 2]]))
        report = component_topology(*join([closed, sheet]))
        self.assertFalse(report['global_closed_edge_topology'])
        self.assertEqual(report['component_count'], 2)
        self.assertEqual(report['closed_component_count'], 1)
        self.assertAlmostEqual(report['closed_face_fraction'], 12/13)
        self.assertEqual(report['boundary_edges'], 3)
        self.assertEqual(sum(x['face_count'] for x in report['components']), 13)

    def test_nonmanifold_component_is_not_accepted_closed(self):
        vertices, faces = box([-.2, -.1, 0], [.2, .1, .3])
        report = component_topology(vertices, np.concatenate([faces, faces[:1]]))
        self.assertFalse(report['global_closed_edge_topology'])
        self.assertEqual(report['closed_component_count'], 0)
        self.assertEqual(report['nonmanifold_edges'], 3)

    def test_locked_prefix_preserves_order_across_scenes(self):
        records = locked_prefix([dict(scene='a', positions=[4, 2], group_ids=[10, 20]),
                                 dict(scene='b', positions=[0], group_ids=[30])], 3)
        self.assertEqual([(x['scene'], x['position'], x['group_id']) for x in records],
                         [('a', 4, 10), ('a', 2, 20), ('b', 0, 30)])

    def test_pose_reader_keeps_only_observation_keys_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'steps.jsonl'
            path.write_text(json.dumps(dict(task_ids=[['a', 10], ['a', 20]], poses=[
                dict(center=[0., 0., 0.], yaw=.1, paired_quality={'unused': 'not geometry'}),
                dict(center=[1., 2., 3.], yaw=.2)]))+'\n', encoding='utf-8')
            poses = recorded_poses(path, 1)
            self.assertEqual(set(poses[0]), {'scene', 'group_id', 'center', 'yaw'})
            self.assertEqual(poses[0]['group_id'], 10)
            with self.assertRaises(ValueError):
                recorded_poses(path, 65)
            with self.assertRaises(ValueError):
                recorded_poses(path, 3)


if __name__ == '__main__':
    unittest.main()

import unittest
import tempfile
from pathlib import Path
import numpy as np
from support_regions import (prepare_scene,prepare_asset,propose,FEATURE_NAMES,
                             save_prepared_scene,load_prepared_scene)


def rectangle(width,depth,step=.01,z=0):
    x,y=np.meshgrid(np.arange(-width/2,width/2+step/2,step),
                    np.arange(-depth/2,depth/2+step/2,step),indexing='ij')
    return np.column_stack((x.ravel(),y.ravel(),np.full(x.size,z)))


class SupportRegionTests(unittest.TestCase):
    def test_orientation_changes_fit(self):
        xyz=rectangle(.45,.70)
        scene=prepare_scene(xyz,'synthetic',pool_size=len(xyz),cell_size=.015)
        asset=prepare_asset(scene,[.60,.35,.2])
        i=np.argmin(np.linalg.norm(xyz[scene['pool_ids'],:2],axis=1))
        self.assertGreater(asset['features'][i,2,2],.9)
        self.assertLess(asset['features'][i,0,2],.8)

    def test_hole_is_not_filled(self):
        xyz=rectangle(1.2,1.2,.02)
        xyz=xyz[np.any(np.abs(xyz[:,:2])>.2,axis=1)]
        scene=prepare_scene(xyz,'hole',pool_size=len(xyz),cell_size=.025)
        self.assertTrue(scene['regions'])
        for r in scene['regions']:
            cell=np.floor((np.zeros(2)-r['origin'])/r['cell']).astype(int)
            if np.all(cell>=0) and np.all(cell<r['mask'].shape):
                self.assertFalse(r['mask'][tuple(cell)])
        self.assertFalse(scene['report']['convex_hull_filled'])

    def test_unique_budget_backup_and_finite(self):
        xyz=rectangle(2,2,.03)
        grounded=dict(scene='s',source='grounding_dino_sam2',point_ids=np.arange(300),
                      label_ids=np.zeros(300,dtype=np.int16),confidence=np.full(300,.9),labels=['floor'])
        scene=prepare_scene(xyz,'s',grounded,pool_size=1024)
        asset=prepare_asset(scene,[.3,.4,.7])
        baseline=np.arange(256)
        ids,f,r=propose(scene,asset,baseline,dict(scene='s',asset_scene='s_a',entry='e'))
        self.assertEqual(len(np.unique(ids)),256)
        self.assertEqual(f.shape,(256,8,len(FEATURE_NAMES)))
        self.assertEqual(r['origin_counts']['global_backup'],32)
        self.assertTrue(np.isin(ids[:32],baseline).all())
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(r['new_point_count'],0)
        self.assertFalse(r['hard_geometry_filter'])

    def test_no_receiver_falls_back_without_false_semantic(self):
        xyz=rectangle(1,1,.02)
        scene=prepare_scene(xyz,'missing',pool_size=1024)
        asset=prepare_asset(scene,[.3,.3,.3])
        ids,f,r=propose(scene,asset,np.arange(256),dict(scene='missing',asset_scene='a',entry='e'))
        self.assertEqual(len(ids),256)
        self.assertFalse(r['grounded_available'])
        self.assertTrue((f[:,:,6]==0).all())

    def test_rejects_gt_receiver_provenance(self):
        xyz=rectangle(1,1,.04)
        with self.assertRaises(ValueError):
            prepare_scene(xyz,'s',dict(scene='s',source='ground_truth',point_ids=np.arange(3),
                           label_ids=np.zeros(3,int),confidence=np.ones(3),labels=['table']))

    def test_scene_cache_preserves_proposals_and_holes(self):
        xyz=rectangle(1.2,1.2,.02)
        xyz=xyz[np.any(np.abs(xyz[:,:2])>.2,axis=1)]
        scene=prepare_scene(xyz,'cache_hole',pool_size=1024)
        args=(np.arange(256),dict(scene='cache_hole',asset_scene='a',entry='e'))
        expected=propose(scene,prepare_asset(scene,[.3,.4,.2]),*args)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.npz'
            save_prepared_scene(path,scene,'identity')
            loaded=load_prepared_scene(path,xyz,'identity')
            actual=propose(loaded,prepare_asset(loaded,[.3,.4,.2]),*args)
            np.testing.assert_array_equal(expected[0],actual[0])
            np.testing.assert_array_equal(expected[1],actual[1])
            with self.assertRaises(ValueError):load_prepared_scene(path,xyz,'wrong_identity')


if __name__=='__main__':unittest.main()

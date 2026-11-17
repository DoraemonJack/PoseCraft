import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from wrapped_collision_v10 import WrappedAssetMeshGeometry,_index,_load_proxy,geometry_digest,sha
from asset_mesh_geometry_v6 import AssetMeshGeometry as OriginalGeometry


def box(lo,hi):
    lo,hi=np.asarray(lo),np.asarray(hi)
    v=np.array([[x,y,z] for z in (lo[2],hi[2]) for y in (lo[1],hi[1]) for x in (lo[0],hi[0])],float)
    f=np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],[0,1,5],[0,5,4],
                [2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]])
    return v,f


def floor(z=0):
    return np.array([[-2,-2,z],[2,-2,z],[2,2,z],[-2,2,z]],float),np.array([[0,1,2],[0,2,3]])


def wall(x):
    return np.array([[x,-.7,-.2],[x,.7,-.2],[x,.7,.7],[x,-.7,.7]]),np.array([[0,1,2],[0,2,3]])


def joined(a,b):
    return np.concatenate((a[0],b[0])),np.concatenate((a[1],b[1]+len(a[0])))


class WrappedGeometryTests(unittest.TestCase):
    original=box([-.25,-.08,0],[.25,.08,.3])
    wrapped=box([-.251,-.081,-.001],[.251,.081,.301])

    def core(self,scene,**kwargs):
        return WrappedAssetMeshGeometry(*scene,*self.original,
            wrapped_vertices=self.wrapped[0],wrapped_faces=self.wrapped[1],**kwargs)

    def test_original_support_is_bitwise_preserved_with_expanded_wrap(self):
        scene=joined(floor(),wall(.15))
        old=OriginalGeometry(*scene,*self.original)
        new=self.core(scene)
        np.testing.assert_array_equal(old.support_samples,new.support_samples)
        for c,y in (([0,0,0],0),([.13,-.02,.019],.4),([0,0,.2],math.pi/2)):
            a,b=old.query(c,y),new.query(c,y)
            for k in ('support_contact_fraction','support_known_fraction','support_gap_fraction',
                      'support_sample_count','support_samples_world','support_distances_m'):
                if isinstance(a[k],np.ndarray):np.testing.assert_array_equal(a[k],b[k],err_msg=k)
                else:self.assertEqual(a[k],b[k],k)

    def test_wrapped_offset_not_recentered_or_rescaled(self):
        core=self.core(floor())
        self.assertEqual(float(core.asset[:,:,2].min()),-.001)
        np.testing.assert_allclose(core.asset.min((0,1)),[-.251,-.081,-.001])
        r=core.query([0,0,0],0)
        self.assertEqual(r['support_contact_fraction'],1.)
        self.assertTrue(r['collision_proxy_original_anchor_preserved'])
        self.assertFalse(r['collision_proxy_recentering_applied'])
        self.assertEqual(r['collision_noncontact_pairs'],0)

    def test_yaw_rotates_wrapped_collision(self):
        core=self.core(wall(.15))
        a,b=core.query([0,0,0],0),core.query([0,0,0],math.pi/2)
        self.assertGreater(a['collision_noncontact_pairs'],0)
        self.assertEqual(b['collision_noncontact_pairs'],0)
        self.assertGreater(a['collision_cost'],b['collision_cost'])
        self.assertAlmostEqual(b['directional_collision_vector'][7],1.)

    def test_collision_is_wrap_geometry_not_original(self):
        scene=wall(.2505)
        old=OriginalGeometry(*scene,*self.original).query([0,0,0],0)
        new=self.core(scene).query([0,0,0],0)
        self.assertEqual(old['collision_noncontact_pairs'],0)
        self.assertGreater(new['collision_noncontact_pairs'],0)
        self.assertEqual(new['collision_cost'],0.)  # inside 20mm tolerance

    def test_open_original_uses_closed_wrap_without_changing_support(self):
        v,f=self.original
        new=WrappedAssetMeshGeometry(*wall(.15),v,f[:-1],
            wrapped_vertices=self.wrapped[0],wrapped_faces=self.wrapped[1])
        self.assertTrue(new.query([0,0,0],0)['asset_closed_edge_topology'])
        self.assertFalse(new.query([0,0,0],0)['collision_proxy_partial_asset'])

    def test_original_hash_is_dtype_stable_not_dimension_only(self):
        v,f=self.original
        self.assertEqual(geometry_digest(v,f),geometry_digest(v.astype('float64'),f.astype('int32')))
        changed=v.copy();changed[0,0]+=.001
        self.assertNotEqual(geometry_digest(v,f),geometry_digest(changed,f))

    def test_changed_wrap_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'wrap.json';source.write_text('{}')
            m=root/'manifest.json'
            row=dict(version='RefinePlace_v1_wrapped_fields',expected_assets=1,
                assets={'x':{'original_geometry_digest':'abc'}},wrapped_manifest_path=str(source),wrapped_manifest_sha256=sha(source))
            m.write_text(json.dumps(row));source.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'manifest changed'):_index(m)

    def test_changed_proxy_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'wrap.json';source.write_text('{}')
            proxy=root/'asset.npz';proxy.write_bytes(b'oldmesh')
            expected=sha(proxy);proxy.write_bytes(b'changedmesh')
            field=root/'field.npz';field.write_bytes(b'field')
            row=dict(asset_uuid='fixture',original_geometry_digest=geometry_digest(*self.original),
                wrapped_path=str(proxy),wrapped_sha256=expected,field_path=str(field),field_sha256=sha(field))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(version='RefinePlace_v1_wrapped_fields',expected_assets=1,
                assets={'x':row},wrapped_manifest_path=str(source),wrapped_manifest_sha256=sha(source))))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):_load_proxy(*self.original,manifest)

    def test_wrapped_geometry_closed_admission_required(self):
        with self.assertRaisesRegex(ValueError,'full closed-edge'):
            WrappedAssetMeshGeometry(*floor(),*self.original,
                wrapped_vertices=self.wrapped[0],wrapped_faces=self.wrapped[1][:-1])


if __name__=='__main__':unittest.main()

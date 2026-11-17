"""Prepare all admitted Alpha Wrap fields once; no labels/scenes are read."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from wrapped_collision_v10 import _prepare_asset,_mesh,geometry_digest,sha,DEFAULT_WRAP


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def prepare_one(item):
    uid,row,output,grid_side,samples=item
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for path,key in ((row['path'],'sha256'),(row['source_path'],'source_sha256')):
        if sha(path)!=row[key]:raise ValueError('Input hash changed '+uid)
    if not row['validation']['topology_pass']:
        raise ValueError('Wrap failed prior topology admission '+uid)
    with np.load(row['source_path'],allow_pickle=False) as z:
        original_digest=geometry_digest(z['vertices'],z['faces'])
    with np.load(row['path'],allow_pickle=False) as z:
        triangles=_mesh(z['vertices'],z['faces'],'wrapped_asset')
    start=time.monotonic()
    data,_=_prepare_asset(triangles,grid_side,samples)
    if not data['closed'] or not data['recovery']['full_surface_coverage']:
        raise ValueError('Unexpected open wrapper '+uid)
    metadata={k:v for k,v in data.items() if not isinstance(v,np.ndarray) and k!='tree'}
    arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    path=Path(output)/(uid+'.npz')
    np.savez_compressed(path,**arrays)
    result=dict(asset_uuid=uid,original_geometry_digest=original_digest,
        source_path=row['source_path'],source_sha256=row['source_sha256'],
        wrapped_path=row['path'],wrapped_sha256=row['sha256'],
        field_path=str(path),field_sha256=sha(path),field_metadata=metadata,
        parameters=row['parameters'],source_dimensions_m=row['source_dimensions_m'],
        coordinate_frame=row['coordinate_frame'],anchor_transform_applied=row['anchor_transform_applied'],
        grid_side=grid_side,surface_samples=samples,preparation_seconds=time.monotonic()-start)
    write(Path(output)/(uid+'.json'),result)
    return uid,result


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.manifest.read_text(encoding='utf-8'))
    if source['failures']:raise ValueError('Wrap manifest contains failed assets')
    expected=source['total_assets'];assert len(source['assets'])==expected
    rows={};started=time.monotonic()
    jobs=[(uid,row,str(a.output),a.grid_side,a.samples) for uid,row in source['assets'].items()]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare_one,x) for x in jobs]):
            uid,result=future.result();rows[uid]=result
            progress=dict(completed=len(rows),total=expected,elapsed_seconds=time.monotonic()-started,
                last_asset=uid,last_resolved=result['field_metadata']['resolved'],
                resolved=sum(r['field_metadata']['resolved'] for r in rows.values()))
            write(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    data=dict(version='RefinePlace_v1_wrapped_fields',expected_assets=expected,assets=rows,
        wrapped_manifest_path=str(a.manifest.resolve()),wrapped_manifest_sha256=sha(a.manifest),
        grid_side=a.grid_side,surface_samples=a.samples,code_sha256=sha(__file__),
        no_annotation_reads=True,original_asset_and_anchor_unchanged=True,
        total_seconds=time.monotonic()-started)
    write(a.output/'manifest.json',data)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=DEFAULT_WRAP)
    p.add_argument('--output',type=Path,default=Path('/path/to/research300/refineplace_v1_0_20261103_v1/wrapped_fields'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--grid-side',type=int,default=64)
    p.add_argument('--samples',type=int,default=4096)
    main(p.parse_args())

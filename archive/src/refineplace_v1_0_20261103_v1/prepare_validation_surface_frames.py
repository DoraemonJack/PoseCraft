"""Four preregistered observation-only frames for each fixed validation scene."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT.parents[1]/'DATA/ScanNet_v2/tasks/scannet_frames_25k.zip'
OUT=ROOT/'validation_grounded_frames'
def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    manifest=ROOT/'fixed_validation150.json'
    scenes=sorted({t['scene'] for t in json.loads(manifest.read_text())['tasks']})
    frames={};files=[];audits=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        names=set(z.namelist())
        for scene in scenes:
            prefix='scannet_frames_25k/'+scene+'/'
            ids={k:set() for k in ('color','depth','pose')}
            for name in names:
                m=re.fullmatch(re.escape(prefix)+r'(color|depth|pose)/(\d+)\.(jpg|png|txt)',name)
                if m:ids[m[1]].add(int(m[2]))
            common=sorted(set.intersection(*ids.values()))
            assert len(common)>=4
            frames[scene]=[common[i] for i in np.rint(np.linspace(0,len(common)-1,4)).astype(int)]
            content={}
            for suffix in ['intrinsics_depth.txt','intrinsics_color.txt']+[f'{kind}/{f:06d}.{ext}' for f in frames[scene] for kind,ext in [('color','jpg'),('depth','png'),('pose','txt')]]:
                data=z.read(prefix+suffix);dest=OUT/'inputs'/scene/suffix
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);content[suffix]=data
                files.append(dict(relative=dest.relative_to(OUT).as_posix(),sha256=sha(data),bytes=len(data),archive_member=prefix+suffix,archive_crc=z.getinfo(prefix+suffix).CRC))
            for f in frames[scene]:
                pose=np.loadtxt(io.BytesIO(content[f'pose/{f:06d}.txt']))
                dep=np.asarray(Image.open(io.BytesIO(content[f'depth/{f:06d}.png'])))
                assert np.isfinite(pose).all() and pose.shape==(4,4) and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5) and np.any(dep>0)
                audits.append(dict(scene=scene,frame=int(f),basic_IO_valid=True))
    report=dict(partition='fixed_validation150',scenes=scenes,frame_ids=frames,frames=4*len(scenes),files=files,
                frame_rule='round(linspace4) sorted common RGB/depth/pose IDs; unchanged after inference',
                audits=audits,task_manifest_sha256=sha(manifest.read_bytes()),source_archive=str(ARCHIVE),
                total_bytes=sum(x['bytes'] for x in files),no_GT_labels=True,
                projection='existing identity color/depth extrinsic approximation, not claimed calibrated')
    (OUT/'frame_source_identity_before_models.json').write_text(json.dumps(report,indent=2))
    with tarfile.open(ROOT/'validation_grounded_frames.tar','w') as tar:
        tar.add(OUT,arcname='validation_grounded_frames')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','audits','frame_ids')},indent=2))

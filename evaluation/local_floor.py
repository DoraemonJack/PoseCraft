"""Local support plane for every on relation.

A failed fit raises FloorFitUnavailable. Callers record that on constraint
as unsatisfied and continue the evaluation.
"""
import numpy as np
from geometry import xy_iom, z_interval
class FloorFitUnavailable(RuntimeError): pass

def floor_on(asset,anchor,points):
 if xy_iom(asset,anchor)<0.5:return False
 if points is None:raise FloorFitUnavailable('anchor points missing')
 p=np.asarray(points,dtype=np.float64);xy=p[:,:2]-asset.center[:2];c=np.cos(asset.yaw);s=np.sin(asset.yaw)
 local=xy @ np.array([[c,-s],[s,c]])
 half=asset.size[:2]/2+0.15
 p=p[(abs(local[:,0])<=half[0]) & (abs(local[:,1])<=half[1])]
 if len(p)<30:raise FloorFitUnavailable('fewer than 30 local support points')
 rng=np.random.default_rng(1729)
 if len(p)>1200:p=p[rng.choice(len(p),1200,replace=False)]
 best=None;cos_limit=np.cos(np.deg2rad(15))
 for _ in range(96):
  a,b,c=p[rng.choice(len(p),3,replace=False)];normal=np.cross(b-a,c-a);norm=np.linalg.norm(normal)
  if norm<1e-10:continue
  normal/=norm
  if abs(normal[2])<cos_limit:continue
  inliers=np.abs((p-a)@normal)<=.02;count=int(inliers.sum())
  if best is None or count>best[0]:best=(count,inliers)
 if best is None or best[0]<30 or best[0]/len(p)<.5:raise FloorFitUnavailable('insufficient RANSAC support')
 q=p[best[1]];centroid=q.mean(axis=0);_,_,vh=np.linalg.svd(q-centroid,full_matrices=False);normal=vh[-1]
 if normal[2]<0:normal=-normal
 if normal[2]<cos_limit:raise FloorFitUnavailable('plane slope exceeds 15 degrees')
 hx,hy=asset.size[:2]/2;local_xy=np.array([[0,0],[-hx,-hy],[-hx,hy],[hx,-hy],[hx,hy]])
 rot=np.array([[np.cos(asset.yaw),-np.sin(asset.yaw)],[np.sin(asset.yaw),np.cos(asset.yaw)]])
 probes=local_xy@rot.T+asset.center[:2];surface=centroid[2]-((probes-centroid[:2])@normal[:2])/normal[2];bottom=z_interval(asset)[0]
 return bool(np.median(np.abs(bottom-surface))<=.01 and np.all(bottom-surface>=-.01))

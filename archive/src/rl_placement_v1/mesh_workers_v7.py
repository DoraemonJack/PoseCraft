"""Persistent spawn workers for unchanged V6 actual-mesh geometry queries.

The GPU-owning parent submits only CPU pose data. Workers retain one scene
adapter/BVH and its 32 asset cores. Exceptions propagate through Future.result;
there is no bounding-box fallback. Only explicit train/val queries are allowed.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path


_WORKER = None
_THREAD_LIMITER = None
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _initialize_worker(base, manifest):
    global _WORKER, _THREAD_LIMITER
    for key in _THREAD_ENV:
        os.environ[key] = '1'
    import torch
    torch.set_num_threads(1)
    # Spawn may import the application's main module before this initializer.
    # Limit already-loaded native BLAS pools as well as future imports.
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _THREAD_LIMITER = None
    from mesh_scene_adapter_v6 import load_mesh_manifest
    _WORKER = dict(base=Path(base), library=load_mesh_manifest(manifest),
                   scene_key=None, adapter=None)


def _worker_query(task):
    from mesh_scene_adapter_v6 import load_scene_mesh, MeshSceneAdapterV6
    if _WORKER is None:
        raise RuntimeError('Geometry worker has not been initialized')
    state = _WORKER
    library = state['library']
    split, scene = task['split'], task['scene']
    allowed = library.manifest['train_scenes' if split == 'train' else 'development_scenes']
    if split not in ('train', 'val') or scene not in allowed:
        raise ValueError('Geometry task split/scene mismatch')
    scene_key = (split, scene)
    if state['scene_key'] != scene_key:
        # Release the previous full scene and BVH before allocating the next.
        state['adapter'], state['scene_key'] = None, None
        xyz, faces = load_scene_mesh(state['base'], scene, split, library)
        state['adapter'] = MeshSceneAdapterV6(xyz, faces, library)
        state['scene_key'] = scene_key
    adapter = state['adapter']
    adapter.set_asset(task['asset_uuid'], task['dims'])
    return adapter.features(task['center'], task['dims'], task['yaw'])


class MeshGeometryService:
    """CPU mesh process pool with order-preserving batch queries.

    ``root`` is the research root containing ``rl_position_adjustment_v1``;
    passing that data directory itself is also accepted. Task fields are
    scene, split (train or val), asset_uuid, dims[3], center[3], yaw radians.
    ``submit`` returns a standard Future of the original (vector,stats,queries).
    """
    def __init__(self, root, mesh_manifest, workers=2):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError('workers must be a positive integer')
        root = Path(root).resolve()
        self.base = root if root.name == 'rl_position_adjustment_v1' else root/'rl_position_adjustment_v1'
        self.mesh_manifest = Path(mesh_manifest).resolve()
        manifest = json.loads(self.mesh_manifest.read_text(encoding='utf-8'))
        if manifest.get('test_reads') != 0:
            raise ValueError('Train/dev-only mesh manifest required')
        self._allowed = {'train': set(manifest['train_scenes']), 'val': set(manifest['development_scenes'])}
        if self._allowed['train'] & self._allowed['val'] or {s[:9] for s in self._allowed['train']} & {s[:9] for s in self._allowed['val']}:
            raise ValueError('Mesh train/development room overlap')
        self.workers = workers
        self._closed = False
        self._executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker,
            initargs=(str(self.base), str(self.mesh_manifest)))

    def _validate_task(self, task):
        if not isinstance(task, dict):
            raise ValueError('Geometry task must be a dictionary')
        required = ('scene', 'split', 'asset_uuid', 'dims', 'center', 'yaw')
        if any(key not in task for key in required):
            raise ValueError('Geometry task requires ' + ', '.join(required))
        split, scene = task['split'], task['scene']
        if split not in self._allowed or scene not in self._allowed[split]:
            raise ValueError('Only matching manifest train/val geometry tasks permitted')
        if not isinstance(task['asset_uuid'], str) or not task['asset_uuid']:
            raise ValueError('asset_uuid must be a nonempty string')
        result = {key: task[key] for key in required}
        for key in ('dims', 'center'):
            try:
                values = tuple(float(value) for value in task[key])
            except (TypeError, ValueError):
                raise ValueError(key + ' must be a finite vector of length three') from None
            if len(values) != 3 or not all(math.isfinite(value) for value in values) or (key == 'dims' and min(values) <= 0):
                raise ValueError(key + ' must contain three finite values; dimensions must be positive')
            result[key] = values
        result['yaw'] = float(task['yaw'])
        if not math.isfinite(result['yaw']):
            raise ValueError('yaw must be finite radians')
        return result

    def submit(self, task):
        if self._closed:
            raise RuntimeError('Geometry service is closed')
        return self._executor.submit(_worker_query, self._validate_task(task))

    def query(self, task):
        return self.submit(task).result()

    def query_many(self, tasks):
        # Validate the complete batch before enqueueing any of its work.
        validated = [self._validate_task(task) for task in tasks]
        if self._closed:
            raise RuntimeError('Geometry service is closed')
        futures = [self._executor.submit(_worker_query, task) for task in validated]
        return [future.result() for future in futures]

    def close(self, wait=True):
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=wait, cancel_futures=True)

    shutdown = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

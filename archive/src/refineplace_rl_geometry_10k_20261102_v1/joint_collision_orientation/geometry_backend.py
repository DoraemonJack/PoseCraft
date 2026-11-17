"""Opt-in process-local collision extension; the original worker files are immutable."""
import os
import sys
from pathlib import Path

WORKERS = Path('/path/to/research100/refineplace_persistent_grouped_pool_modules_v2')


def install_geometry():
    import mesh_scene_adapter_v6 as adapter
    from joint_asset_geometry_v1 import AssetMeshGeometry
    adapter.AssetMeshGeometry = AssetMeshGeometry


def initialize_collision_worker():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(WORKERS))
    import private_worker_geometry_v1 as private
    install_geometry()
    private.initialize()


class CollisionGeometryBackend:
    def __init__(self, workers=12):
        from cpu_pool_backend_v1 import verify_sources
        verify_sources()
        sys.path.insert(0, str(WORKERS))
        import persistent_grouped_geometry_v2 as grouped
        previous = grouped.initialize_process
        try:
            grouped.initialize_process = initialize_collision_worker
            self.service = grouped.PersistentGroupedGeometryService(workers=workers, chunk_size=0)
        finally:
            grouped.initialize_process = previous

    def query_many(self, tasks):
        return self.service.query_many(tasks)

    def close(self):
        self.service.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

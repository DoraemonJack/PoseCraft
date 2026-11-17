"""Use exactly the existing val150 exporter/container with the combined model."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
import evaluate_epoch150_v1 as original_eval
sys.path.insert(0, str(HERE))
from geometry_backend import install_geometry
from orientation_policy_v1 import DirectionalReferencePosePolicy
import joint_rollout_v1 as rollout


def main(a):
    a.output = a.output.resolve()
    a.checkpoint = a.checkpoint.resolve()
    a.training_protocol = a.training_protocol.resolve()
    original_factory = original_eval.d.joint.ReferencePosePolicy
    original_rollout = original_eval.d.joint.rollout_batch
    import mesh_scene_adapter_v6 as adapter_module
    original_geometry = adapter_module.AssetMeshGeometry

    def wrapped_factory(*args, **kwargs):
        return DirectionalReferencePosePolicy(original_factory(*args, **kwargs))

    original_eval.d.joint.ReferencePosePolicy = wrapped_factory
    original_eval.d.joint.rollout_batch = rollout.rollout_batch
    try:
        install_geometry()
        original_eval.main(a)
    finally:
        original_eval.d.joint.ReferencePosePolicy = original_factory
        original_eval.d.joint.rollout_batch = original_rollout
        adapter_module.AssetMeshGeometry = original_geometry
    if not a.smoke and not a.official_smoke:
        d = original_eval.d
        control_path = HERE.parent / 'eval_high_geometry_val150' / 'report.json'
        control = d.read(control_path)
        joint = d.read(a.output / 'report.json')
        assert control['manifest_sha256'] == joint['manifest_sha256']
        assert control['tasks'] == joint['tasks'] == 150
        assert control['counts'] == joint['counts']
        protocol = d.read(a.training_protocol)
        assert control['checkpoint_sha256'] == protocol['control_checkpoint_sha256']
        keys = ('complete_placement_success', 'language_adherence_success', 'global_constraint_accuracy')
        comparison = dict(protocol='original_vs_all_changes_together', tasks=150,
            control_report=str(control_path), joint_report=str(a.output / 'report.json'),
            manifest_sha256=joint['manifest_sha256'], original={k: control['official'][k] for k in keys},
            joint={k: joint['official'][k] for k in keys},
            percentage_point_changes={k: joint['official'][k] - control['official'][k] for k in keys},
            original_constraints=control['official'].get('per_constraint_metrics'),
            joint_constraints=joint['official'].get('per_constraint_metrics'),
            limitations='development validation; same pre-existing reference coverage and official visibility pipeline')
        d.write(HERE / 'comparison.json', comparison)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--training-protocol', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=114)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--official-smoke', action='store_true')
    main(p.parse_args())

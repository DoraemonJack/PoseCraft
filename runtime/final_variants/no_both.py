"""Execute the verified adaptive protocol on one fixed shard of the test set."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path('/path/to/research300/posecraft_adaptive_stopping_20261114_nmax4')
OUT = Path('/path/to/research300/posecraft_terra_parse_epoch8_20261116')
import sys
sys.path.insert(0,str(OUT))
from negative_distance import wrap_negative
DATA = Path('/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1')
TRAIN = Path('/path/to/research300/refineplace_v2_main_retrain_20261103_v1')
SOURCE = Path('/path/to/research300/refineplace_orientation_trust_v19_20261103_v1')
CHECKPOINT_SHA = '03e0f5d3cfc18749b87a0ecd7a176770a3d5cb31d3093a25042c5df8f0a98c44'
EPSILON_ACT, EPSILON_ROT, MAX_STEPS = None, None, 14
parser = argparse.ArgumentParser()
parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
parser.add_argument('--resume', action='store_true')
cli = parser.parse_args()
ids = list(range(cli.shard * 1750, (cli.shard + 1) * 1750))
output = OUT / 'n14_rot8_full_and_ablations_20261117' / 'no_both' / ('shard' + str(cli.shard))

spec = importlib.util.spec_from_file_location('frozen_inference', SOURCE/'infer_retrained_pair_fulltest.py')
runtime = importlib.util.module_from_spec(spec)
source_text = (SOURCE/'infer_retrained_pair_fulltest.py').read_text()
assert "len(saved['path']) > 2" in source_text and 'len(refinement) > 2' in source_text
source_text = source_text.replace("len(saved['path']) > 2", "len(saved['path']) > MAX_STEPS").replace('len(refinement) > 2', 'len(refinement) > MAX_STEPS')
runtime.__dict__['MAX_STEPS'] = MAX_STEPS
runtime.__dict__['wrap_negative'] = wrap_negative
key = "            scorer, pair_info = scorer_factory.for_task(task['prompt'], refs[(task['scene'], task['group_id'])], inferred)"
assert key in source_text
source_text = source_text.replace(key, key + "\n            scorer = wrap_negative(scorer, refs[(task['scene'], task['group_id'])], row['centers'], ctx['candidate_mask'])")
exec(compile(source_text, str(SOURCE/'infer_retrained_pair_fulltest.py'), 'exec'), runtime.__dict__)
assert runtime.sha(TRAIN/'rank0/epoch8.pt') == CHECKPOINT_SHA
runtime.selected_entry_ids = lambda limit, offset=0: ids
old_write = runtime.write


def write(path, value):
    if Path(path).name in ('protocol_before_forward.json', 'summary.json'):
        value.update(adaptive_protocol=dict(epsilon_act=EPSILON_ACT, epsilon_rot=EPSILON_ROT,
                     maximum_steps=MAX_STEPS, stop_semantics='original policy and orientation selection; no gain-based epsilon stopping',
                     action_file_sha256=runtime.sha(HERE/'adaptive_rollout.py'),
                     orientation_file_sha256=runtime.sha(HERE/'adaptive_refiner.py'),
                     wrapper_sha256=runtime.sha(Path(__file__)), negative_distance_sha256=runtime.sha(OUT/'negative_distance.py'), negative_gap_margin_m=0.5, negative_logit_strength=2.0),
                     checkpoint_scope='epoch-8 without pair head; manuscript checkpoint identity unverified')
    old_write(path, value)


runtime.write = write
old_load = runtime.load_runtime
def load_adaptive(data_root):
    import sys
    result=old_load(data_root)
    sys.path.insert(0,str(SOURCE))
    import refiner_v19
    text=(SOURCE/'refiner_v19.py').read_text()
    assert '0 <= max_steps <= 2' in text
    text=text.replace('0 <= max_steps <= 2','0 <= max_steps <= 8')
    exec(compile(text,str(SOURCE/'refiner_v19.py'),'exec'),refiner_v19.__dict__)
    original_rollout=result.rollout.rollout_batch
    original_refine=refiner_v19.refine_after_baseline_v19
    def rollout(*args,**kwargs):
        kwargs['steps']=0
        return original_rollout(*args,**kwargs)
    def refine(*args,**kwargs):
        kwargs['max_steps']=0
        return original_refine(*args,**kwargs)
    result.rollout.rollout_batch=rollout
    refiner_v19.refine_after_baseline_v19=refine
    return result

runtime.load_runtime = load_adaptive
args = argparse.Namespace(checkpoint=TRAIN/'rank0/epoch8.pt', training_protocol=TRAIN/'protocol_before_training.json',
    output=output, pair_head=None, data_root=DATA, references=OUT/'catalog/reference_catalog_compact_v4_test.jsonl',
    references_summary=OUT/'catalog/manifest.json',
    grounded_root=DATA/'receivers/test_grounded_support/grounded_surfaces', baseline_predictions=None,
    v11_baseline_predictions=None, limit=len(ids), offset=ids[0], resume=cli.resume, preflight_only=False)
runtime.main(args)

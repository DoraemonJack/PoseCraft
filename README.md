# PoseCraft

Research code associated with **PoseCraft: Geometry-Aware Pose Refinement for Language-Guided 3D Object Placement**.

## Contents

- `evaluation/`: the custom evaluation implementation used by the final experiments, including the local-plane `on` check, geometry, visibility, aggregation, and the original batch runner. Defaults: near threshold 0.10 times the horizontal room diagonal and facing angle 20 degrees. `run_trial.py` applies the paper's batch configuration.
- `runtime/final_variants/`: recorded scripts for full model, no-action, no-orientation, and neither refinements. The full-model snapshot applies 14 action steps and 8 orientation steps.
- `runtime/reference_ablation/`: reference-masking ablation snapshot.
- `runtime/refiner/`, `runtime/orientation/`, `runtime/placement/`, `runtime/final/`: model inference, refinement, grounding and proposal runtime snapshots.
- `runtime/validation/`: validation inference, grounding and evaluation snapshots.
- `runtime/language_parsing_prompt.txt`: the language parsing prompt supplied with the manuscript.
- `archive/src/`: archived model, loss, training, geometry and support code; directory names preserve their historical versions. Files were selected from the November 9 research archive and its statically identified Python dependencies. This is not a complete inventory of every historical experiment.
- `SOURCE_PROVENANCE.json`: original file hashes before path normalization and the list of files modified for packaging.

## Evaluation

Install the evaluation dependencies in a Python environment:

```bash
python -m pip install -r evaluation/requirements.txt
python evaluation/evaluate.py --help
```

A typical invocation, after supplying the required dataset and scene files, is:

```bash
python evaluation/evaluate.py val \
  --predictions /path/to/predictions.json \
  --h5 /path/to/placeit3d_val.h5 \
  --scans /path/to/scans \
  --assets /path/to/assets \
  --output /path/to/output \
  --near-fraction 0.10 --details
```

Additional annotation paths and alignment metadata may be needed for a given dataset layout; see `--help` and `evaluation/data.py`. The output `--details` is useful for inspecting individual decisions. Physical and visibility computations require the relevant meshes and scene annotations. The CLI and imports were checked locally; the example above has not been executed on a fresh dataset installation.

**Protocol:** this evaluator is a reimplementation, not the released PlaceIt3D official Docker evaluator. Its physical and `on` results differ substantially from the official evaluator on the 297-task historical validation subset. Treat reported results as being under the paper's own evaluation protocol; do not assume numerical equivalence to the official implementation.

## Training and inference snapshots

The runtime drivers preserve the structure of the runs rather than presenting a tested one-command pipeline. Placeholders under `/path/to/research300`, `/path/to/research100`, and `/path/to/home` replace the original machine paths; configure the required paths and dependencies before running. The final full-model driver loads the epoch-8 checkpoint with SHA-256 `03e0f5d3cfc18749b87a0ecd7a176770a3d5cb31d3093a25042c5df8f0a98c44`, then applies runtime wrappers to extend the archived orientation refiner to eight steps. Exact correspondence of every asset and dependent file to the reported full run has not been independently reconstructed from this package. Source hashes in the provenance file refer to files **before** path normalization.

The archive contains multiple historical variants. Do not substitute a variant or checkpoint for the final run solely because its name is similar. Dataset and model access must follow the respective upstream terms. Verify rights and licensing for each component and choose a repository license before public release.

## Input and credentials

Supply data and third-party checkpoints separately. Never commit keys, passwords, private host information, datasets, or generated experiment caches. External detector and segmentation models, their configuration files, and dataset acquisition instructions must be added according to their original project licenses and access terms.

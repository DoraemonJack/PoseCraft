"""Immutable identities for the frozen image/projection stack used by v1.0."""
from pathlib import Path
import json
from build_grounded_surfaces import HERE,MODEL_ROOT,sha,write

if __name__=='__main__':
    paths=[MODEL_ROOT/'dino_base/model.safetensors',MODEL_ROOT/'dino_base/config.json',
           MODEL_ROOT/'dino_base/preprocessor_config.json',MODEL_ROOT/'dino_base/tokenizer.json',
           MODEL_ROOT/'sam2_original_checkpoint.pt',
           Path('/path/to/research100/strict_real30_image_modules_20261102_v1/image_reference_projection_v1.py'),
           HERE/'build_grounded_surfaces.py',HERE/'build_validation_grounded_surfaces.py']
    records=[dict(path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    write(HERE/'grounded_model_provenance.json',dict(source='frozen_DINO_base_official_SAM2_hiera_small',files=records))
    print(json.dumps({'files':len(records),'status':'complete'}))

"""Evaluate checkpoints with the fixed Direction4/6 comparison protocol."""

import argparse
import hashlib
import json
import os
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['ULTRALYTICS_SAFE_LOAD'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--name', default='unified_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    from experiment_notes import write_evaluation_note
    from ultralytics import YOLO
    output = ROOT / 'evaluations' / args.name
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(data=str(ROOT / 'configs/visdrone2019-det.yaml'), split='val', imgsz=640,
                    batch=8, device=args.device, workers=0, conf=0.001, iou=0.7, max_det=300,
                    rect=True, quantize=None, augment=False, agnostic_nms=False, end2end=True,
                    plots=True, save_json=False, project=str(output), exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    write_evaluation_note(output)
    for i, path in enumerate(args.models):
        path = path.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model = YOLO(path)
        assert len(model.names) == 10, 'Expected a 10-class VisDrone checkpoint'
        metrics = model.val(**settings, name=f'{i:02d}_{path.parent.parent.name}')
        result = {'checkpoint': str(path), 'sha256': digest,
                  'metrics': {k: float(v) for k, v in metrics.results_dict.items()},
                  'classes': metrics.summary(decimals=8), 'speed_ms': metrics.speed}
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        (output / f'{i:02d}_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()), encoding='utf-8')
        write_evaluation_note(output)
    print(f'Unified evaluation saved to {output}')


if __name__ == '__main__':
    main()

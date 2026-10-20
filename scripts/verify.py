"""Verify the isolated source, VisDrone labels, and YOLO26n model construction."""

from __future__ import annotations

import argparse
import os
os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "configs" / "visdrone2019-det.yaml"
MODEL_YAML = PROJECT_ROOT / "configs" / "models" / "yolo26n_direction46.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def calculate_flops(model, get_flops, imgsz: int = 640) -> float:
    """Calculate GFLOPs, falling back to full-image THOP for older THOP builds."""
    flops = get_flops(model, imgsz=imgsz)
    if flops:
        return flops
    import thop

    image = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
    macs, _ = thop.profile(model, inputs=[image], verbose=False)
    return macs * 2 / 1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Parse every value in every train/val label.")
    parser.add_argument("--include-test", action="store_true", help="Also check the converted test-dev labels.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny 3-epoch pause/resume regression, not a formal experiment.")
    return parser.parse_args()


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def split_directories(dataset_root: Path, split_value: str) -> tuple[Path, Path]:
    images = dataset_root / split_value
    if images.name != "images":
        raise ValueError(f"Expected an images directory in split path, got: {images}")
    return images, images.with_name("labels")


def validate_split(dataset_root: Path, split_name: str, split_value: str, nc: int, full: bool) -> None:
    image_dir, label_dir = split_directories(dataset_root, split_value)
    images = index_files(image_dir, IMAGE_SUFFIXES)
    labels = index_files(label_dir, {".txt"})
    missing = images.keys() - labels.keys()
    orphaned = labels.keys() - images.keys()
    if not images or missing or orphaned:
        raise RuntimeError(f"{split_name}: missing labels={len(missing)}, orphan labels={len(orphaned)}")

    if not full:
        print(f"{split_name}: {len(images):,} images and {len(labels):,} matching labels")
        return

    counts: Counter[int] = Counter()
    boxes = 0
    for label_path in labels.values():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 values")
            values = [float(value) for value in fields]
            class_id = int(values[0])
            if values[0] != class_id or not 0 <= class_id < nc:
                raise ValueError(f"{label_path}:{line_number}: invalid class {values[0]}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label_path}:{line_number}: non-finite value")
            x, y, width, height = values[1:]
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"{label_path}:{line_number}: invalid box {values[1:]}")
            counts[class_id] += 1
            boxes += 1
    print(f"{split_name}: {len(images):,} images, {boxes:,} boxes, classes={dict(sorted(counts.items()))}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if config.get("channels") != 3:
        raise ValueError("VisDrone config must contain channels: 3")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    split_names = ["train", "val"]
    if args.include_test:
        split_names.append("test")
    for split_name in split_names:
        validate_split(dataset_root, split_name, config[split_name], len(config["names"]), args.full)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics import __file__ as package_file, __version__
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    source = Path(package_file).resolve()
    if PROJECT_ROOT.resolve() not in source.parents:
        raise RuntimeError(f"Imported a non-project Ultralytics package: {source}")
    checked = check_det_dataset(str(DATA_YAML.resolve()), autodownload=False)
    model = DetectionModel(str(MODEL_YAML), ch=checked["channels"], nc=checked["nc"], verbose=False).eval()
    with torch.no_grad():
        model(torch.zeros(1, checked["channels"], 64, 64))

    parameters = sum(parameter.numel() for parameter in model.parameters())
    flops = calculate_flops(model, get_flops)
    print(f"Isolated YOLO26 verification passed: Ultralytics {__version__} at {source}")
    print(f"Input channels={checked['channels']}, classes={checked['names']}, end2end={model.end2end}")
    print(f"Direction-4+6 model: parameters={parameters:,}, GFLOPs@640={flops:.3f}, strides={model.stride.tolist()}")

    verify_fusion(args.smoke)

def verify_fusion(smoke=False):
    """Verify fusion, initialization and the resume regression without a formal experiment."""
    import copy
    import csv
    import hashlib
    import json
    import os
    import time
    import cv2
    import thop
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel, torch_safe_load
    from ultralytics.utils.torch_utils import init_seeds

    torch.set_num_threads(4)
    config = yaml.safe_load(DATA_YAML.read_text(encoding='utf-8'))
    pretrained = YOLO(PROJECT_ROOT / 'weights/yolo26n.pt').model
    imgroot = Path(config['path']) / config['train']
    imgs, rows = [], []
    for i, path in enumerate(sorted(imgroot.glob('*.jpg'))[:2]):
        im = cv2.imread(str(path))
        assert im is not None
        imgs.append(torch.from_numpy(cv2.cvtColor(cv2.resize(im, (640, 640)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        for line in (path.parent.with_name('labels') / (path.stem + '.txt')).read_text().splitlines():
            if line.strip():
                rows.append((i, list(map(float, line.split()))))
    batch = {'img': torch.stack(imgs).float() / 255,
             'batch_idx': torch.tensor([i for i, _ in rows]),
             'cls': torch.tensor([[r[0]] for _, r in rows]),
             'bboxes': torch.tensor([r[1:] for _, r in rows])}
    report = {'formal_training_started': False, 'models': {}}
    common = None
    for tag, filename in [('F0', 'yolo26n_direction6.yaml'), ('F1', 'yolo26n_direction46.yaml')]:
        init_seeds(0, deterministic=True)
        model = DetectionModel(str(PROJECT_ROOT / 'configs/models' / filename), nc=10, verbose=False)
        model.names = config['names']
        model.load(pretrained)
        initial = {k: v.detach().clone() for k, v in model.state_dict().items() if '.bifpn.' not in k}
        if common is None:
            common = initial
        else:
            assert common.keys() == initial.keys()
            assert all(torch.equal(v, initial[k]) for k, v in common.items()), 'F0/F1 common initialization differs'
        model.eval()
        assert model.stride.tolist() == [8, 16]
        params = sum(p.numel() for p in model.parameters())
        macs, _ = thop.profile(copy.deepcopy(model), inputs=[torch.zeros(1, 3, 640, 640)], verbose=False)
        with torch.no_grad():
            before = model(batch['img'][:1])[0]
            rect = model(torch.zeros(1, 3, 384, 640))[0]
            assert before.shape[-1] == 6 and torch.isfinite(before).all() and torch.isfinite(rect).all()
            original = model(batch['img'][:1])[1]['one2one']
            fused = copy.deepcopy(model).fuse(verbose=False)
            after = fused(batch['img'][:1])[1]['one2one']
            for k in ('boxes', 'scores'):
                torch.testing.assert_close(after[k], original[k], atol=3e-4, rtol=3e-4)
        checkpoint = PROJECT_ROOT / 'research' / f'{tag}_roundtrip.pt'
        torch.save({'model': copy.deepcopy(model)}, checkpoint)
        restored = torch_safe_load(checkpoint, safe_only=True)[0]['model'].eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(batch['img'][:1])[0], before)
        model.args = get_cfg(overrides={'epochs': 300, 'nwd_weight': 0.2, 'nwd_constant': 12.8, 'nwd_max_size': 32.0})
        model.train()
        loss, _ = model(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        edges = [p.grad.abs().sum().item() for n, p in model.named_parameters() if 'edge_weights' in n]
        assert all(x > 0 for x in edges)
        gpu = model.cuda()
        gpu.criterion = None
        gpu.zero_grad(set_to_none=True)
        b8 = {'img': batch['img'].repeat(4, 1, 1, 1).cuda(),
              'batch_idx': torch.cat([batch['batch_idx'] + i * 2 for i in range(4)]).cuda(),
              'cls': batch['cls'].repeat(4, 1).cuda(), 'bboxes': batch['bboxes'].repeat(4, 1).cuda()}
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.float16):
            amp_loss, _ = gpu(b8)
        amp_loss.sum().backward()
        assert torch.isfinite(amp_loss).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters() if p.requires_grad)
        peak = torch.cuda.max_memory_allocated() / 2**20
        gpu.cpu()
        del b8, restored, gpu
        torch.cuda.empty_cache()
        report['models'][tag] = {'parameters': params, 'gflops_thop_640': macs * 2 / 1e9,
            'strides': model.stride.tolist(), 'edge_grad_l1': edges, 'amp_batch': 8, 'amp_peak_allocated_mb': peak,
            'checks': ['common_initialization', 'rectangular_inference', 'fused_inference', 'restricted_roundtrip',
                       'real_image_loss_backward', 'amp_loss_backward']}
        print(tag, report['models'][tag], flush=True)
    if smoke:
        report['resume_smoke'] = verify_resume()
    manifests = json.loads((PROJECT_ROOT / 'research/source_manifest.json').read_text())
    for root, files in manifests.items():
        for rel, digest in files.items():
            assert hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() == digest, (root, rel)
    report['original_sources_unchanged'] = True
    (PROJECT_ROOT / 'research/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def verify_resume():
    """Train 19 images briefly, pause after save, then assert checkpoint state before resumed training."""
    import csv
    import json
    from ultralytics import YOLO
    from ultralytics.nn.tasks import torch_safe_load
    from ultralytics.utils.torch_utils import init_seeds

    project = PROJECT_ROOT / 'runs_smoke'
    from datetime import datetime
    name = 'fusion_resume_regression_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    run = project / name
    assert not run.exists(), 'Do not overwrite an existing smoke test'
    init_seeds(0, deterministic=True)
    model = YOLO(PROJECT_ROOT / 'configs/models/yolo26n_direction46.yaml')
    model.load(PROJECT_ROOT / 'weights/yolo26n.pt')

    class SavedPause(Exception):
        pass

    def pause(trainer):
        assert trainer.epoch == 0 and trainer.last.is_file()
        raise SavedPause('Intentional pause after epoch 1 checkpoint; smoke only')

    model.add_callback('on_model_save', pause)
    try:
        model.train(data=str(DATA_YAML), project=str(project), name=name, epochs=3, imgsz=640,
                    batch=8, workers=0, device='0', fraction=0.003, optimizer='SGD', seed=0,
                    lr0=0.01, lrf=0.01, nwd_weight=0.2, nwd_constant=12.8, nwd_max_size=32.,
                    close_mosaic=0, amp=True, plots=False, deterministic=True)
        raise AssertionError('Expected intentional pause')
    except SavedPause:
        pass
    old_csv = (run / 'results.csv').read_bytes()
    ckpt = torch_safe_load(run / 'weights/last.pt', safe_only=True)[0]
    assert ckpt['epoch'] == 0 and ckpt['optimizer'] is not None
    assert isinstance(ckpt['train_args']['pretrained'], (str, Path)), 'Must test original pretrained-path resume bug'
    expected = {k: v.float().clone() for k, v in ckpt['ema'].state_dict().items()}
    checks = {}

    def restored(trainer):
        assert trainer.start_epoch == 1
        assert trainer.args.workers == 0 and trainer.args.nwd_weight == 0.2
        actual = trainer.model.state_dict()
        assert all(torch.equal(v.to(actual[k].device), actual[k]) for k, v in expected.items())
        current = trainer.optimizer.state_dict()
        assert current['param_groups'] == ckpt['optimizer']['param_groups']
        for key, vals in ckpt['optimizer']['state'].items():
            for k, v in vals.items():
                target = current['state'][key][k]
                assert torch.equal(v.to(target.device), target) if isinstance(v, torch.Tensor) else v == target
        assert trainer.scaler.state_dict() == ckpt['scaler']
        assert trainer.ema.updates == ckpt['updates']
        checks.update(model_equal=True, optimizer_equal=True, scaler_equal=True, ema_updates_equal=True,
                      start_epoch_1based=2, pretrained_path_preserved=str(trainer.args.pretrained))

    resumed = YOLO(run / 'weights/last.pt')
    resumed.add_callback('on_train_start', restored)
    resumed.train(resume=True, device='0')
    assert (run / 'results.csv').read_bytes().startswith(old_csv)
    rows = list(csv.DictReader((run / 'results.csv').open()))
    assert [int(r['epoch']) for r in rows] == [1, 2, 3]
    assert checks.get('model_equal')
    checks.update(epochs=[1, 2, 3], csv_prefix_unchanged=True, formal_training=False)
    (PROJECT_ROOT / 'research/resume_regression.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return checks


if __name__ == "__main__":
    main()

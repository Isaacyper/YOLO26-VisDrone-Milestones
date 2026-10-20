"""Verify the isolated source, VisDrone labels, and YOLO26n model construction."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "configs" / "visdrone2019-det.yaml"
MODEL_YAML = PROJECT_ROOT / "configs" / "models" / "yolo26n_baseline.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Parse every value in every train/val label.")
    parser.add_argument("--include-test", action="store_true", help="Also check the converted test-dev labels.")
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
    if not images:
        raise RuntimeError(f"{split_name}: no images found in {image_dir}")
    if missing or orphaned:
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

    source = Path(package_file).resolve()
    if PROJECT_ROOT.resolve() not in source.parents:
        raise RuntimeError(f"Imported a non-project Ultralytics package: {source}")
    checked = check_det_dataset(str(DATA_YAML.resolve()), autodownload=False)
    model = DetectionModel(str(MODEL_YAML), ch=checked["channels"], nc=checked["nc"], verbose=False).eval()
    with torch.no_grad():
        model(torch.zeros(1, checked["channels"], 64, 64))

    verify_bifpn(config)

    print(f"Isolated YOLO26 verification passed: Ultralytics {__version__} at {source}")
    print(f"Input channels={checked['channels']}, classes={checked['names']}, end2end={model.end2end}")


def verify_bifpn(config):
    """Exercise real-image loss/backward, AMP, transfer, serialization and fused inference without training a run."""
    import copy
    import hashlib
    import io
    import json
    import cv2
    import thop
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(4)
    torch.manual_seed(0)
    results = {}
    pretrained = YOLO(PROJECT_ROOT / "weights/yolo26n.pt").model
    image_dir = Path(config["path"]) / config["train"]
    images, labels = [], []
    for path in sorted(image_dir.glob("*.jpg"))[:2]:
        im = cv2.imread(str(path))
        if im is None:
            raise RuntimeError(f"Unreadable image: {path}")
        images.append(torch.from_numpy(cv2.cvtColor(cv2.resize(im, (640, 640)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        labels.append([list(map(float, line.split())) for line in path.with_name(path.stem + ".txt").parent.with_name("labels").joinpath(path.stem + ".txt").read_text().splitlines() if line.strip()])
    assert len(images) == 2 and all(labels), "Need two annotated images for loss validation"
    rows = [(i, row) for i, lines in enumerate(labels) for row in lines]
    batch = {"img": torch.stack(images).float() / 255, "batch_idx": torch.tensor([i for i, _ in rows]),
             "cls": torch.tensor([[r[0]] for _, r in rows]), "bboxes": torch.tensor([r[1:] for _, r in rows])}
    for path in sorted((PROJECT_ROOT / "configs/models").glob("*.yaml")):
        model = DetectionModel(str(path), nc=10, verbose=False).eval()
        # Match the normal YOLO trainer: names are attached before transfer.
        model.names = config["names"]
        model.load(pretrained)
        compatible = {k: v for k, v in pretrained.state_dict().items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape and '.cv3.' not in k and '.one2one_cv3.' not in k}
        assert all(torch.equal(model.state_dict()[k], v) for k, v in compatible.items())
        params = sum(x.numel() for x in model.parameters())
        # Same full-image THOP fallback used in Direction6. FLOPs = 2 * MACs; estimate excludes some functional ops.
        macs, _ = thop.profile(copy.deepcopy(model), inputs=[torch.zeros(1, 3, 640, 640)], verbose=False)
        with torch.no_grad():
            pred = model(batch["img"][:1])[0]
            assert torch.isfinite(pred).all() and pred.shape[-1] == 6
            rect = model(torch.zeros(1, 3, 384, 640))[0]
            assert torch.isfinite(rect).all()
        buffer = io.BytesIO()
        torch.save({"model": model}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)["model"].eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(batch["img"][:1])[0], pred)
            fused = copy.deepcopy(model).fuse(verbose=False)
            original_raw = model(batch["img"][:1])[1]["one2one"]
            fused_raw = fused(batch["img"][:1])[1]["one2one"]
            for key in ("boxes", "scores"):
                torch.testing.assert_close(fused_raw[key], original_raw[key], atol=2e-4, rtol=2e-4)
        model.args = get_cfg(overrides={"epochs": 300})
        model.train()
        loss, _ = model(batch)
        loss.sum().backward()
        grads = [v.grad for v in model.parameters() if v.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), "Missing or nonfinite gradients"
        edge_grads = [v.grad.abs().sum().item() for n, v in model.named_parameters() if "edge_weights" in n]
        assert all(v > 0 for v in edge_grads), "Fusion weights did not receive gradients"
        amp_checked = False
        if torch.cuda.is_available() and path.stem == "yolo26n_direction4":
            model.zero_grad(set_to_none=True)
            model = model.cuda()
            model.criterion = None
            gpu_batch = {"img": batch["img"].repeat(4, 1, 1, 1).cuda(),
                         "cls": batch["cls"].repeat(4, 1).cuda(),
                         "bboxes": batch["bboxes"].repeat(4, 1).cuda(),
                         "batch_idx": torch.cat([batch["batch_idx"] + 2 * i for i in range(4)]).cuda()}
            with torch.autocast("cuda", dtype=torch.float16):
                amp_loss, _ = model(gpu_batch)
            amp_loss.sum().backward()
            assert torch.isfinite(amp_loss).all()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            amp_checked = True
            model.cpu()
            del gpu_batch
            torch.cuda.empty_cache()
        results[path.stem] = {"parameters": params, "gflops_thop_640": macs * 2 / 1e9,
            "strides": model.stride.tolist(), "compatible_nonclass_tensors": len(compatible),
            "loss": loss.detach().tolist(), "edge_grad_l1": edge_grads, "cuda_amp_checked": amp_checked,
            "checks": ["pretrained_transfer", "640_inference", "rectangular_inference", "checkpoint_roundtrip", "fused_inference", "real_image_loss_backward"]}
        print(path.stem, results[path.stem], flush=True)
    from ultralytics.nn.modules.head import BiFPNRefine
    bridge = BiFPNRefine((64, 128), width=32).train()
    inputs = [torch.randn(2, 64, 80, 80, requires_grad=True), torch.randn(2, 128, 40, 40, requires_grad=True)]
    outputs = bridge(inputs)
    assert [x.shape for x in outputs] == [x.shape for x in inputs]
    sum(x.square().mean() for x in outputs).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)
    original = Path(r"D:/ALSS/YOLO26-VisDrone2019-DET")
    manifest = json.loads((PROJECT_ROOT / "research/source_manifest.json").read_text())
    assert all(hashlib.sha256((original / k).read_bytes()).hexdigest() == v for k, v in manifest.items()), "Original source changed"
    report = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "original_source_unchanged": True,
              "formal_training_started": False, "main_amp_batch": 8, "two_level_fusion_interface_checked": True, "models": results}
    (PROJECT_ROOT / "research/verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

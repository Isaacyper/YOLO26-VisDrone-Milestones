"""Benchmark baseline and Direction-6 YOLO26 forward latency on one device."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "baseline": PROJECT_ROOT / "configs" / "models" / "yolo26n_baseline.yaml",
    "direction6": PROJECT_ROOT / "configs" / "models" / "yolo26n_direction6.yaml",
    "direction46": PROJECT_ROOT / "configs" / "models" / "yolo26n_direction46.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA before reading wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("--imgsz, --iterations and --repeats must be positive; --warmup must be non-negative")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics.nn.tasks import DetectionModel

    device = torch.device(args.device)
    image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    print(f"device={device}, batch=1, imgsz={args.imgsz}, iterations={args.iterations}, repeats={args.repeats}")
    with torch.inference_mode():
        for name, model_yaml in MODELS.items():
            model = DetectionModel(str(model_yaml), ch=3, nc=10, verbose=False).eval().fuse(verbose=False).to(device)
            for _ in range(args.warmup):
                model(image)
            timings = []
            for _ in range(args.repeats):
                synchronize(device)
                start = time.perf_counter()
                for _ in range(args.iterations):
                    model(image)
                synchronize(device)
                timings.append((time.perf_counter() - start) * 1000 / args.iterations)
            latency = statistics.median(timings)
            print(
                f"{name:<11} {latency:>7.3f} ms/image, {1000 / latency:>7.1f} FPS "
                f"(range {min(timings):.3f}-{max(timings):.3f} ms)"
            )


if __name__ == "__main__":
    main()

"""Compare baseline and Direction-6 YOLO26 model complexity at 640 pixels."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "baseline": PROJECT_ROOT / "configs" / "models" / "yolo26n_baseline.yaml",
    "direction6": PROJECT_ROOT / "configs" / "models" / "yolo26n_direction6.yaml",
    "direction46": PROJECT_ROOT / "configs" / "models" / "yolo26n_direction46.yaml",
}


def calculate_flops(model, get_flops, imgsz: int = 640) -> float:
    """Calculate GFLOPs, falling back to full-image THOP for older THOP builds."""
    flops = get_flops(model, imgsz=imgsz)
    if flops:
        return flops
    import thop

    image = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
    macs, _ = thop.profile(model, inputs=[image], verbose=False)
    return macs * 2 / 1e9


def main() -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    results = {}
    for name, model_yaml in MODELS.items():
        model = DetectionModel(str(model_yaml), ch=3, nc=10, verbose=False).eval()
        results[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "gflops": calculate_flops(model, get_flops),
            "strides": model.stride.tolist(),
        }

    baseline = results["baseline"]
    direction6 = results["direction46"]
    print("model       parameters    GFLOPs@640    strides")
    for name, values in results.items():
        print(f"{name:<11} {values['parameters']:>10,}    {values['gflops']:>10.3f}    {values['strides']}")
    print(
        "Direction-4+6 change vs baseline: "
        f"parameters {(direction6['parameters'] / baseline['parameters'] - 1) * 100:+.1f}%, "
        f"GFLOPs {(direction6['gflops'] / baseline['gflops'] - 1) * 100:+.1f}%"
    )


if __name__ == "__main__":
    main()

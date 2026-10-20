"""Train this project's local YOLO26 source on VisDrone2019-DET."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "configs" / "visdrone2019-det.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26n.pt"
DEFAULT_RUNS = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(PROJECT_ROOT / "configs/models/yolo26n_direction4.yaml"), help="Checkpoint or YOLO26 model YAML.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Optional transfer weights for a YAML model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--name", default="b2_direction4_bifpn_300e")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved training options without training.")
    return parser.parse_args()


def load_local_yolo():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics import YOLO, __file__ as package_file, __version__

    source = Path(package_file).resolve()
    if PROJECT_ROOT.resolve() not in source.parents:
        raise RuntimeError(f"Imported a non-project Ultralytics package: {source}")
    print(f"Using local Ultralytics {__version__}: {source}")
    return YOLO


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise ValueError("--epochs and --imgsz must be positive, and --batch cannot be zero")
    if args.workers < 0 or args.patience < 0 or args.close_mosaic < 0:
        raise ValueError("--workers, --patience, and --close-mosaic must be non-negative")
    if not 0.0 <= args.mosaic <= 1.0:
        raise ValueError("--mosaic must be in [0, 1]")

    YOLO = load_local_yolo()
    os.chdir(PROJECT_ROOT)
    options = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        amp=args.amp,
        cache=args.cache,
        fraction=args.fraction,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        plots=True,
    )
    if args.dry_run:
        import json
        print(json.dumps({"model": args.model, "weights": args.weights, "train": options}, indent=2))
        return
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    model = YOLO(args.model)
    if args.weights and not args.resume and Path(args.model).suffix.lower() in {".yaml", ".yml"}:
        if not Path(args.weights).is_file():
            raise FileNotFoundError(args.weights)
        model.load(args.weights)
    model.train(**options)


if __name__ == "__main__":
    main()

"""Train this project's local YOLO26 source on VisDrone2019-DET."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "configs" / "visdrone2019-det.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "configs" / "models" / "yolo26n_direction46_single_dysample.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26n.pt"
DEFAULT_RUNS = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Checkpoint or YOLO26 model YAML.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Optional transfer weights for a YAML model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--nwd-weight", type=float, default=0.2)
    parser.add_argument("--nwd-constant", type=float, default=12.8)
    parser.add_argument("--nwd-max-size", type=float, default=32.0)
    parser.add_argument("--box-iou", choices=["ciou"], default="ciou")
    parser.add_argument("--name", default="ds_f2_dysample_300e_seed0")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_runtime():
    """Keep generated runtime files inside this independent experiment directory."""
    for key, folder in {
        "YOLO_CONFIG_DIR": "ultralytics",
        "MPLCONFIGDIR": "matplotlib",
        "TORCH_HOME": "torch",
        "TEMP": "tmp",
        "TMP": "tmp",
    }.items():
        path = PROJECT_ROOT / ".local" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def attach_run_controls(model):
    """Record provenance and pause only after a complete epoch checkpoint is saved."""
    import hashlib
    import json
    import subprocess

    def provenance(trainer):
        output = trainer.save_dir / "source_manifest.json"
        entries = json.loads((PROJECT_ROOT / "datasets_local/manifest.json").read_text(encoding="utf-8"))
        changed_data = [
            entry["local"]
            for entry in entries
            if not Path(entry["local"]).is_file()
            or hashlib.sha256(Path(entry["local"]).read_bytes()).hexdigest() != entry["sha256"]
        ]
        if changed_data:
            raise RuntimeError(f"Dataset changed; inspect before training or resuming: {changed_data[:10]}")
        if not output.exists():
            paths = [
                *PROJECT_ROOT.glob("scripts/*.py"),
                *PROJECT_ROOT.glob("configs/**/*.yaml"),
                *PROJECT_ROOT.glob("ultralytics/**/*.py"),
                PROJECT_ROOT / "ultralytics/cfg/default.yaml",
                *PROJECT_ROOT.glob("*.ps1"),
            ]
            state = {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
                ).strip(),
                "files": {
                    p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "git_branch": subprocess.check_output(
                    ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
                ).strip(),
                "git_dirty": subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
                ).strip(),
                "paper_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "papers/DySample_ICCV2023.pdf").read_bytes()
                ).hexdigest(),
                "data_manifest_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "datasets_local/manifest.json").read_bytes()
                ).hexdigest(),
                "initial_weights_sha256": hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest(),
            }
            output.write_text(json.dumps(state, indent=2), encoding="utf-8")
        else:
            state = json.loads(output.read_text(encoding="utf-8"))
            changed = [
                p
                for p, h in state["files"].items()
                if not (PROJECT_ROOT / p).is_file() or hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() != h
            ]
            if changed:
                raise RuntimeError(f"Source changed since this run started; inspect before resuming: {changed}")
            current_hashes = {
                "paper_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "papers/DySample_ICCV2023.pdf").read_bytes()
                ).hexdigest(),
                "data_manifest_sha256": hashlib.sha256(
                    (PROJECT_ROOT / "datasets_local/manifest.json").read_bytes()
                ).hexdigest(),
                "initial_weights_sha256": hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest(),
            }
            if any(state.get(k) != v for k, v in current_hashes.items()):
                raise RuntimeError("Paper, data manifest or initialization weights changed since the run started")

    def pause(trainer):
        if (trainer.save_dir / "PAUSE").exists():
            print(f"PAUSED after epoch {trainer.epoch + 1}. Resume checkpoint: {trainer.last}")
            raise KeyboardInterrupt("Requested pause after complete checkpoint; remove PAUSE before resume")

    model.add_callback("on_train_start", provenance)
    model.add_callback("on_model_save", pause)


def load_local_yolo():
    configure_runtime()
    os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
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
    if not 0.0 <= args.nwd_weight <= 1.0:
        raise ValueError("--nwd-weight must be in [0, 1]")
    if args.nwd_constant <= 0.0 or args.nwd_max_size <= 0.0:
        raise ValueError("--nwd-constant and --nwd-max-size must be positive")

    from experiment_notes import attach_training_notes

    YOLO = load_local_yolo()
    os.chdir(PROJECT_ROOT)
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    if args.resume:
        # Do not silently override checkpoint settings with argparse defaults.
        permitted = {"--model", "--resume", "--device", "--dry-run"}
        unexpected = {s.split("=", 1)[0] for s in sys.argv[1:] if s.startswith("--")} - permitted
        if unexpected:
            raise ValueError(f"Resume preserves checkpoint settings; remove {sorted(unexpected)}")
        if Path(args.model).suffix.lower() != ".pt":
            raise ValueError("Resume requires an explicit last.pt checkpoint")
        from ultralytics.nn.tasks import torch_safe_load

        checkpoint, _ = torch_safe_load(args.model, safe_only=True)
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            raise ValueError("Checkpoint is finalized or missing optimizer; cannot resume")
        if checkpoint.get("training_state") is None:
            raise ValueError("This experiment requires full raw-model training_state for resume")
        if args.dry_run:
            print(
                {
                    "resume": str(Path(args.model).resolve()),
                    "next_epoch": checkpoint["epoch"] + 2,
                    "saved_settings": checkpoint["train_args"],
                }
            )
            return
        pause_file = Path(checkpoint["train_args"]["save_dir"]) / "PAUSE"
        if pause_file.exists():
            raise ValueError(f"Remove the PAUSE marker before resuming: {pause_file}")
        model = YOLO(args.model)
        attach_training_notes(model)
        attach_run_controls(model)
        model.train(resume=True, device=args.device)
        return
    if (args.project / args.name).exists():
        raise FileExistsError(f"Run already exists: {args.project / args.name}; choose a new name or resume last.pt")
    if args.weights and not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)
    if args.dry_run:
        import json

        print(json.dumps(vars(args), default=str, indent=2))
        return
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(args.seed, deterministic=True)
    model = YOLO(args.model)
    if args.weights and not args.resume and Path(args.model).suffix.lower() in {".yaml", ".yml"}:
        model.load(args.weights)

    attach_training_notes(model)
    attach_run_controls(model)
    model.train(
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
        box_iou=args.box_iou,
        nwd_weight=args.nwd_weight,
        nwd_constant=args.nwd_constant,
        nwd_max_size=args.nwd_max_size,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        val=args.val,
        plots=True,
    )


if __name__ == "__main__":
    main()

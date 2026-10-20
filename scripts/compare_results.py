"""Audit training protocols and summarize same-row best validation metrics; never launches training."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = "epochs patience batch imgsz optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr seed deterministic amp fraction rect cos_lr close_mosaic multi_scale freeze box cls cls_pw dfl nbs hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr mosaic mixup cutmix copy_paste max_det iou conf end2end single_cls classes".split()
METRICS = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

def dataset_signature(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: data.get(k) for k in ("path", "train", "val", "channels", "names")}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, default=[], help="Additional run directory (repeatable).")
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    protocol.update(epochs=300, patience=100, batch=8, imgsz=640, optimizer="SGD", seed=0, deterministic=True)
    expected_data = dataset_signature(ROOT / "configs/visdrone2019-det.yaml")
    weight_hash = hashlib.sha256((ROOT / "weights/yolo26n.pt").read_bytes()).hexdigest()
    runs = [Path(r"D:/ALSS/runs_yolo26_visdrone/yolo26n_visdrone_sgd_100e"),
            Path(r"D:/ALSS/YOLO26-VisDrone2019-DET-Direction6/runs/yolo26n_visdrone_direction6"),
            Path(r"D:/ALSS/YOLO26-VisDrone2019-DET-Direction6-PaperFaithful/runs/a2_direction6_strict_nwd_300e")]
    runs += sorted((ROOT / "runs").iterdir()) + args.run
    rows = []
    for run in dict.fromkeys(runs):
        if not run.is_dir() or not (run / "args.yaml").exists():
            continue
        cfg = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        differences = {k: {"expected": protocol.get(k), "actual": cfg.get(k)} for k in FIELDS if cfg.get(k) != protocol.get(k)}
        if dataset_signature(cfg["data"]) != expected_data:
            differences["dataset"] = "different split, root, channels or class mapping"
        weight = cfg.get("pretrained")
        if not isinstance(weight, str) or not Path(weight).is_file():
            differences["initial_weights"] = "checkpoint identity not verified"
        elif hashlib.sha256(Path(weight).read_bytes()).hexdigest() != weight_hash:
            differences["initial_weights"] = "different checkpoint SHA256"
        result = {"run": str(run), "scheduled_epochs": cfg["epochs"], "protocol_differences": differences,
                  "nwd_weight": cfg.get("nwd_weight", 0), "nwd_max_size": cfg.get("nwd_max_size"),
                  "status": "no results", "metrics": None}
        csv_path = run / "results.csv"
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                history = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
            if history:
                best = max(history, key=lambda r: float(r[METRICS[-1]]))
                result.update(status="schedule matches" if not differences else "historical / protocol differs",
                              completed_epochs=int(float(history[-1]["epoch"])), best_epoch=int(float(best["epoch"])),
                              metrics={k: float(best[k]) for k in METRICS})
        rows.append(result)
    report = {"note": "Metrics are from the single CSV row with maximum val mAP50-95, not a fresh best.pt evaluation. NWD is an intentional treatment difference. Matching protocol does not prove convergence or source-version equality.", "runs": rows}
    out = ROOT / "research"
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Cross-direction comparison", "", report["note"], "", "| Run | Completed / cap | Best epoch | P | R | mAP50 | mAP50-95 | Protocol |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        values = [f"{row['metrics'][k] * 100:.3f}" for k in METRICS] if row["metrics"] else ["-"] * 4
        lines.append(f"| {Path(row['run']).name} | {row.get('completed_epochs', 0)} / {row['scheduled_epochs']} | {row.get('best_epoch', '-')} | " + " | ".join(values) + f" | {row['status']} |")
        if row["protocol_differences"]:
            print(Path(row["run"]).name, row["protocol_differences"])
    lines += ["", "Values above are percentages. Direction4 has no trained accuracy until its experiments finish.", "See comparison.json for every protocol difference. Training cap is 300 with patience 100; early stopping remains enabled."]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()

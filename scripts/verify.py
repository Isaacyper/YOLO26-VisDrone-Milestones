"""Verify F2 + DySample, initialization, real-image gradients and checkpoint resume."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from train import PROJECT_ROOT as ROOT, attach_run_controls, configure_runtime, load_local_yolo

configure_runtime()
YOLO = load_local_yolo()
import torch
import torch.nn.functional as F
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import DySample
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.torch_utils import init_seeds, strip_optimizer

MODELS = {
    "control": ROOT / "configs/models/yolo26n_direction46_single.yaml",
    "dysample": ROOT / "configs/models/yolo26n_direction46_single_dysample.yaml",
}
PARAMETERS = {"control": 605812, "dysample": 612020}
DATA = ROOT / "configs/visdrone2019-det.yaml"
WEIGHTS = ROOT / "weights/yolo26n.pt"


def equal(a, b):
    """Assert nested saved training states match exactly across devices."""
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def verify_module():
    """Compare to fixed upstream LP and test zero offsets and differentiability."""
    spec = importlib.util.spec_from_file_location("upstream_dysample", ROOT / "research/upstream/dysample.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    for channels, h, w in [(64, 4, 7), (128, 5, 3)]:
        x = torch.randn(2, channels, h, w, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        local = DySample(channels)
        reference = upstream.DySample(channels, style="lp", groups=4, dyscope=False)
        reference.load_state_dict(local.state_dict())
        output = local(x)
        expected = reference(y)
        equal(output, expected)
        output.square().mean().backward()
        expected.square().mean().backward()
        equal(x.grad, y.grad)
        equal(local.offset.weight.grad, reference.offset.weight.grad)
        assert torch.isfinite(x.grad).all() and local.offset.weight.grad.abs().sum() > 0
        with torch.no_grad():
            local.offset.weight.zero_()
            local.offset.bias.zero_()
        torch.testing.assert_close(
            local(x), F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), atol=2e-6, rtol=2e-6
        )
    return {
        "upstream_fp32_forward_exact": True,
        "upstream_cpu_backward_exact": True,
        "zero_offset_matches_bilinear": True,
        "offset_gradients_nonzero": True,
    }


def verify_initialization():
    """Check shared F2 parameters and the global RNG after model initialization."""
    models = {}
    rng = {}
    for variant, file in MODELS.items():
        init_seeds(0, deterministic=True)
        models[variant] = YOLO(file).model
        rng[variant] = torch.get_rng_state()
    base = models["control"].state_dict()
    candidate = models["dysample"].state_dict()
    for k, v in base.items():
        equal(v, candidate[k])
    equal(rng["control"], rng["dysample"])
    for variant, m in models.items():
        assert sum(p.numel() for p in m.parameters()) == PARAMETERS[variant]
        assert m.stride.tolist() == [8.0, 16.0]
    assert isinstance(models["dysample"].model[10], DySample)
    assert isinstance(models["dysample"].model[13], DySample)
    for m in models.values():
        m.load(YOLO(WEIGHTS).model)
    base = models["control"].state_dict()
    for k, v in base.items():
        equal(v, models["dysample"].state_dict()[k])
    return {
        "common_initial_parameters_exact": True,
        "common_parameters_after_transfer_exact": True,
        "global_rng_preserved": True,
        "parameters": PARAMETERS,
        "detection_strides": [8, 16],
        "extra_parameters": 6208,
    }


def verify_loss():
    """Ensure CIoU plus NWD is identical to the historical F2 loss owner."""
    source = Path(json.loads((ROOT / "research/f2_source_snapshot.json").read_text())["source"])
    spec = importlib.util.spec_from_file_location(
        "ultralytics.utils.historical_f2_loss", source / "ultralytics/utils/loss.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 1.0, 4.0, 4.0], [0.0, 0.0, 40.0, 40.0], [4.0, 4.0, 8.0, 8.0]])
    target = pred + torch.tensor([0.2, 0.1, 0.4, 0.3])
    inputs = [
        torch.ones(1, 4, 4),
        pred.unsqueeze(0),
        torch.ones(4, 2),
        target.unsqueeze(0),
        torch.ones(1, 4, 10),
        torch.tensor(40.0),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([640, 640]),
        torch.ones(4, 1) * 8,
    ]
    equal(BboxLoss(1, nwd_weight=0.2, box_iou="ciou")(*inputs), old.BboxLoss(1, nwd_weight=0.2)(*inputs))
    return {"ciou_nwd_matches_historical_f2_exactly": True}


def verify_backward():
    """Test the same real-image batch under FP32/AMP for both variants."""
    import cv2

    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    images, targets = [], []
    for i, p in enumerate(sorted((Path(cfg["path"]) / cfg["train"]).glob("*.jpg"))[:2]):
        image = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (640, 640)), cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(image).permute(2, 0, 1))
        for line in (p.parent.with_name("labels") / p.with_suffix(".txt").name).read_text().splitlines():
            if line.strip():
                targets.append((i, list(map(float, line.split()))))
    batch = {
        "img": torch.stack(images).float().repeat(4, 1, 1, 1).cuda() / 255,
        "batch_idx": torch.cat([torch.tensor([i for i, _ in targets]) + 2 * j for j in range(4)]).cuda(),
        "cls": torch.tensor([[v[0]] for _, v in targets]).repeat(4, 1).cuda(),
        "bboxes": torch.tensor([v[1:] for _, v in targets]).repeat(4, 1).cuda(),
    }
    report = {}
    for variant, file in MODELS.items():
        init_seeds(0, deterministic=True)
        model = YOLO(file).load(WEIGHTS).model.cuda().train()
        model.args = get_cfg(overrides={"epochs": 300, "box_iou": "ciou", "nwd_weight": 0.2})
        item = {}
        for amp in (False, True):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, _ = model(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad
            )
            row = {
                "loss": float(loss.detach().sum()),
                "all_unscaled_gradients_finite": True,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            if variant == "dysample":
                row["offset_gradient_l1"] = {
                    str(i): float(model.model[i].offset.weight.grad.abs().sum()) for i in [10, 13]
                }
                assert all(v > 0 for v in row["offset_gradient_l1"].values())
            item["amp" if amp else "fp32"] = row
        if variant == "dysample":
            shapes = {}
            handles = [
                model.model[i].register_forward_hook(lambda m, a, b, i=i: shapes.update({str(i): list(b.shape)}))
                for i in [10, 13]
            ]
            model.eval()
            with torch.no_grad():
                prediction = model(torch.zeros(1, 3, 640, 640, device="cuda"))[0]
                assert torch.isfinite(prediction).all()
            for h in handles:
                h.remove()
            assert shapes == {"10": [1, 128, 80, 80], "13": [1, 64, 160, 160]}
            model.half()
            with torch.no_grad():
                assert torch.isfinite(model(torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16))[0]).all()
            item["neck_shapes_640"] = shapes
            item["fp16_inference_finite"] = True
        report[variant] = item
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {"batch": 8, "source_images": 2, "repeat_factor": 4, "imgsz": 640, "variants": report}


def tiny_data():
    """Create independent copies of 16 train/8 val images for workflow checks only."""
    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    original = Path(cfg["path"])
    root = ROOT / ".local/smoke_data"
    for split, count in [("train", 16), ("val", 8)]:
        for image in sorted((original / cfg[split]).glob("*.jpg"))[:count]:
            for src, dst in [
                (image, root / split / "images" / image.name),
                (
                    image.parent.with_name("labels") / image.with_suffix(".txt").name,
                    root / split / "labels" / image.with_suffix(".txt").name,
                ),
            ]:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
        cfg[split] = f"{split}/images"
    cfg["path"] = root.as_posix()
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def release(model):
    """Close short-run loaders and free their device memory."""
    for key in ("train_loader", "test_loader"):
        loader = getattr(model.trainer, key, None)
        if loader is not None and hasattr(loader, "close"):
            loader.close()
    model.trainer = None
    gc.collect()
    torch.cuda.empty_cache()


def smoke(variant):
    """Check complete saved state and a real second epoch for both model variants."""
    from experiment_notes import attach_training_notes

    init_seeds(0, deterministic=True)
    name = f"ds_smoke_{variant}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    model = YOLO(MODELS[variant]).load(WEIGHTS)
    attach_training_notes(model)
    attach_run_controls(model)
    model.add_callback("on_train_start", lambda t: (t.save_dir / "PAUSE").touch())
    try:
        model.train(
            data=str(tiny_data()),
            project=str(ROOT / "runs_smoke"),
            name=name,
            epochs=4,
            imgsz=640,
            batch=8,
            workers=0,
            device="0",
            optimizer="SGD",
            seed=0,
            lr0=0.01,
            lrf=0.01,
            nwd_weight=0.2,
            nwd_constant=12.8,
            nwd_max_size=32.0,
            box_iou="ciou",
            close_mosaic=0,
            amp=True,
            plots=False,
            deterministic=True,
        )
        raise AssertionError("Expected safe pause")
    except KeyboardInterrupt:
        pass
    run = ROOT / "runs_smoke" / name
    checkpoint = run / "weights/last.pt"
    ckpt = torch_safe_load(checkpoint, safe_only=True)[0]
    assert ckpt["epoch"] == 0 and ckpt["optimizer"] is not None
    state = ckpt["training_state"]
    assert state is not None and state["criterion"]["updates"] == 1
    equal(state["model"], model.trainer.model.state_dict())
    equal(state["ema"], model.trainer.ema.ema.state_dict())
    equal(ckpt["optimizer"], model.trainer.optimizer.state_dict())
    assert sum(p.numel() for p in model.trainer.model.parameters()) == PARAMETERS[variant]
    result = {
        "run": str(run),
        "training_batches_per_epoch": 2,
        "parameters": PARAMETERS[variant],
        "saved_raw_model_exact": True,
        "saved_ema_exact": True,
        "saved_optimizer_exact": True,
        "amp": bool(model.trainer.amp),
        "amp_scaler_after_epoch1": ckpt["scaler"],
        "nonfinite_scaled_gradient_tensors": sum(
            not bool(torch.isfinite(v).all()) for v in state["gradients"].values()
        ),
    }
    release(model)
    old_csv = (run / "results.csv").read_bytes()
    (run / "PAUSE").unlink()
    restored = {}
    resumed = YOLO(checkpoint)
    attach_training_notes(resumed)
    attach_run_controls(resumed)

    def check(t):
        assert t.start_epoch == 1
        equal(t.model.state_dict(), state["model"])
        equal(t.ema.ema.state_dict(), state["ema"])
        equal(t.optimizer.state_dict(), ckpt["optimizer"])
        equal(t.scaler.state_dict(), ckpt["scaler"])
        equal(t.model.criterion.state_dict(), state["criterion"])
        equal(t.scheduler.state_dict(), state["scheduler"])
        equal(vars(t.stopper), state["stopper"])
        assert t.last_opt_step == state["last_opt_step"] and t.accumulate == state["accumulate"]
        for k, p in t.model.named_parameters():
            gradient = state["gradients"].get(k)
            if gradient is None:
                assert p.grad is None
            else:
                equal(p.grad, gradient)
        assert t.ema.updates == ckpt["updates"]
        assert (
            t.args.box_iou == "ciou"
            and t.args.batch == 8
            and t.args.workers == 0
            and t.args.epochs == 4
            and t.args.seed == 0
        )
        equal(torch.get_rng_state(), state["torch_rng"])
        equal(torch.cuda.get_rng_state_all(), state["cuda_rng"])
        restored.update(
            raw_model=True,
            ema=True,
            optimizer=True,
            scaler=True,
            criterion=True,
            scheduler=True,
            early_stopping=True,
            pending_gradients=True,
            optimizer_step_counter=True,
            rng=True,
            settings=True,
        )
        (t.save_dir / "PAUSE").touch()

    resumed.add_callback("on_train_start", check)
    try:
        resumed.train(resume=True, device="0")
        raise AssertionError("Expected second safe pause")
    except KeyboardInterrupt:
        pass
    after = torch_safe_load(checkpoint, safe_only=True)[0]
    assert after["epoch"] == 1 and after["training_state"]["criterion"]["updates"] == 2
    assert (run / "results.csv").read_bytes().startswith(old_csv)
    assert [int(r["epoch"]) for r in csv.DictReader((run / "results.csv").open())] == [1, 2]
    result.update(restored_state_checks=restored, epochs=[1, 2], csv_prefix_preserved=True)
    t = resumed.trainer
    for p in t.model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 0.0001)
    t.accumulate = 8
    t.last_opt_step = 0
    t.save_model()
    pending = torch_safe_load(checkpoint, safe_only=True)[0]
    assert pending["training_state"]["gradients"]
    BaseTrainer._load_checkpoint_state(t, pending)
    for k, p in t.model.named_parameters():
        if p.requires_grad:
            equal(p.grad, pending["training_state"]["gradients"][k])
    t.optimizer.step()
    expected = {k: v.detach().clone() for k, v in t.model.state_dict().items()}
    expected_optimizer = copy.deepcopy(t.optimizer.state_dict())
    BaseTrainer._load_checkpoint_state(t, pending)
    t.optimizer.step()
    equal(t.model.state_dict(), expected)
    equal(t.optimizer.state_dict(), expected_optimizer)
    result["nonempty_pending_gradients_next_optimizer_step_exact"] = True
    stripped = run / "weights/stripped_check.pt"
    shutil.copy2(checkpoint, stripped)
    final = strip_optimizer(stripped)
    assert final["optimizer"] is None and final["training_state"] is None and final["epoch"] == -1
    loaded = YOLO(stripped).model.eval().cuda()
    with torch.no_grad():
        assert torch.isfinite(loaded(torch.zeros(1, 3, 640, 640, device="cuda"))[0]).all()
    result["finalized_model_roundtrip_inference_finite"] = True
    result["finalized_checkpoint_not_resumable"] = True
    assert not list((run / "weights").glob("*.tmp"))
    result["atomic_checkpoint_replace_completed"] = True
    from experiment_notes import write_training_note

    write_training_note(run, "两轮暂停/恢复检查通过；包含合成梯度诊断，不作精度实验")
    release(resumed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    init_seeds(0, deterministic=True)
    report = {
        "formal_training_started": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "module": verify_module(),
        "initialization": verify_initialization(),
        "loss": verify_loss(),
        "real_image_backward": verify_backward(),
    }
    if args.smoke:
        report["control_smoke"] = smoke("control")
        report["dysample_resume_smoke"] = smoke("dysample")
    snapshot = json.loads((ROOT / "research/f2_source_snapshot.json").read_text(encoding="utf-8"))
    for rel, digest in snapshot["source_files"].items():
        assert hashlib.sha256((Path(snapshot["source"]) / rel).read_bytes()).hexdigest() == digest, rel
    report["old_f2_sources_unchanged"] = True
    report["cuda_grid_sample_backward_bitwise_determinism_guaranteed"] = False
    (ROOT / "research/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Generate human-readable experiment sidecars without changing results or checkpoints."""

from pathlib import Path
from datetime import datetime
import csv
import json
import yaml

KEY = "metrics/mAP50-95(B)"
PARAMS = [
    ("box_iou", "框回归IoU损失"),
    ("wiou_alpha", "WIoU alpha"),
    ("wiou_delta", "WIoU delta"),
    ("wiou_momentum", "WIoU动态均值动量"),
    ("epochs", "训练轮数上限"),
    ("seed", "随机种子"),
    ("imgsz", "输入尺寸"),
    ("batch", "批大小"),
    ("workers", "数据加载进程数"),
    ("optimizer", "优化器"),
    ("lr0", "初始学习率"),
    ("lrf", "最终学习率比例"),
    ("momentum", "动量"),
    ("weight_decay", "权重衰减"),
    ("warmup_epochs", "预热轮数"),
    ("patience", "早停耐心"),
    ("amp", "混合精度"),
    ("deterministic", "确定性设置"),
    ("fraction", "数据比例"),
    ("mosaic", "Mosaic"),
    ("close_mosaic", "关闭Mosaic的最后轮数"),
    ("nwd_weight", "NWD混合权重"),
    ("nwd_constant", "NWD归一化常数"),
    ("nwd_max_size", "NWD尺度门限"),
    ("pretrained", "配置中的预训练字段"),
    ("resume", "配置中的恢复字段"),
]


def identity(name, folder):
    if name.startswith(("ds_f2_control", "ds_smoke_control")):
        return (
            "D0：F2同工程对照",
            "F2单层BiFPN，Neck两处最近邻上采样，CIoU＋小目标NWD。",
            "与D1采用同一训练和保存流程，隔离DySample上采样作用。",
        )
    if name.startswith(("ds_f2_dysample", "ds_smoke_dysample")):
        return (
            "D1：F2＋DySample",
            "只将F2第10和13层上采样替换为DySample LP（2倍、4组、静态0.25），CIoU＋NWD。",
            "验证Neck动态采样能否改善定位和小目标精度；短测试不用于精度结论。",
        )
    if name.startswith(("d8_f2_control", "d8_wiou_smoke_ciou")):
        return (
            "W0：F2＋CIoU对照",
            "F2单层BiFPN，CIoU＋小目标NWD，与W1采用同一训练和检查点流程。",
            "隔离WIoU的影响；历史F2为额外参照，不替代同流程对照。",
        )
    if name.startswith("d8_f2_wiou") or "d8_wiou_smoke" in name:
        return (
            "W1：F2＋WIoU v3",
            "保持F2结构，以WIoU v3替换CIoU，小目标NWD混合权重0.2不变。alpha=1.9，delta=3；共享两分支动态均值。",
            "检验框定位精度能否提高；结构与推理参数量不增加。短轮数流程检查不用于精度结论。",
        )
    if name.startswith("f2_"):
        return (
            "F2：方向四＋方向六，单层BiFPN",
            "方向六混合版＋64通道、重复1次、残差0.1的P3/P4加权BiFPN。",
            "仅将F1的BiFPN重复次数2改为1，比较F0/F1，判断减轻融合是否有效。",
        )
    if name.startswith("f1_") or name.startswith("fusion_resume"):
        return (
            "F1：方向四＋方向六，双层BiFPN",
            "方向六混合版＋64通道、重复2次、残差0.1的P3/P4加权BiFPN。",
            "与F0比较融合模块收益，并与原版B0、方向四B2比较精度和轻量化取舍。",
        )
    if name.startswith("f0_"):
        return (
            "F0：方向六混合版对照",
            "FCM/MKP轻量骨干、P2参与融合、P3/P4检测，不添加方向四BiFPN。",
            "为F1/F2提供同配置对照。历史方向六workers4，本组workers0。",
        )
    if name.startswith("b0_") or name.startswith("baseline_") or name.startswith("yolo26n_visdrone_sgd"):
        return (
            "B0/原版基准",
            "原始YOLO26n，P3/P4/P5检测。",
            "作为精度与复杂度基准；100轮历史实验不能直接替代300轮对照。",
        )
    if name.startswith("b1_"):
        return (
            "B1：方向四单层加权BiFPN",
            "原版YOLO26n＋64通道、重复1次、残差0.1的P3/P4/P5加权BiFPN。",
            "与B0和B2比较，考察融合深度的影响。",
        )
    if name.startswith("b2_"):
        return (
            "B2：方向四双层加权BiFPN",
            "原版YOLO26n＋64通道、重复2次、残差0.1的P3/P4/P5加权BiFPN。",
            "方向四主方案；与B0配对种子比较收益，与B1/B3比较深度及加权机制。",
        )
    if name.startswith("b3_"):
        return (
            "B3：方向四双层等权BiFPN",
            "方向四双层融合，使用等权平均而非可学习边权。",
            "与B2比较，隔离可学习融合权重的贡献。",
        )
    if "strict_nwd" in name:
        return (
            "A2：方向六论文严格版",
            "论文严格版FCM/MKP、P3/P4检测；NWD全目标替换CIoU项，保留YOLO26另一回归项。",
            "研究严格结构＋完整NWD的效果；与混合版同时存在结构及损失差异，不能单独归因。",
        )
    if "direction6" in name:
        return (
            "方向六混合版",
            "FCM/MKP轻量骨干、P2融合、P3/P4检测；小目标NWD混合。",
            "检验大幅减少参数时能保留多少精度，为后续融合提供轻量结构。",
        )
    return "待人工命名的实验", "结构以实际model配置为准，未根据名称猜测。", "目的尚未登记，请补充后再用于正式对照。"


def fmt(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_training_note(folder, phase=None):
    folder = Path(folder)
    args = yaml.safe_load((folder / "args.yaml").read_text(encoding="utf-8-sig")) or {}
    snapshot = folder.name.startswith("backup")
    name = folder.parent.name if snapshot else folder.name
    title, structure, purpose = identity(name, folder)
    smoke = "smoke" in str(folder).lower() or "resume_regression" in name
    rows = []
    if (folder / "results.csv").exists():
        with (folder / "results.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items() if k is not None}
                if row.get(KEY) and row.get("epoch"):
                    rows.append(row)
    best = max(rows, key=lambda x: float(x[KEY])) if rows else None
    n = int(rows[-1]["epoch"]) if rows else 0
    target = args.get("epochs")
    if n and target and n >= target:
        status = f"已记录{n}/{target}轮"
    elif n:
        status = f"已记录{n}/{target}轮；不足轮数上限，需结合早停/运行状态判断"
        if best and args.get("patience") and n - int(best["epoch"]) >= args["patience"]:
            status += "（轮数符合早停条件）"
    else:
        status = "有配置但尚无有效results.csv；可能尚未完成首轮或为遗留目录"
    if phase:
        status += f"；入口回调状态：{phase}"
    kind = "备份快照，不是独立实验" if snapshot else ("流程检查，不用于精度结论" if smoke else "正式实验/候选实验")
    lines = [
        f"# {title}",
        "",
        f"- 目录：`{folder}`",
        f"- 类型：**{kind}**",
        f"- 状态：{status}",
        f"- 说明更新时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 内容与目的",
        "",
        structure,
        "",
        purpose,
        "",
        "以下配置读取自本目录args.yaml。该文件可能已被续训更新，不一定等于首次启动配置。",
        "",
        f"- model：`{args.get('model', '未记录')}`",
        f"- data：`{args.get('data', '未记录')}`",
        "",
        "|配置|实际记录|",
        "|---|---|",
    ]
    lines += [f"|{label}|{fmt(args.get(key, '未记录'))}|" for key, label in PARAMS]
    if (folder / "source_manifest.json").exists():
        provenance = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
        lines += [
            "",
            f"代码版本：`{provenance['git_commit']}`；分支：`{provenance.get('git_branch')}`；启动时未提交修改：`{provenance.get('git_dirty') or '无'}`。",
        ]
    lines += ["", "## 结果摘要", "", "本节来自训练CSV，指标为百分比，不与独立FP32验证结果混作同一口径。"]
    if best:
        lines += [
            "",
            f"最佳mAP50–95位于第{best['epoch']}轮；本次说明读取到第{n}轮。",
            "",
            "|指标|最佳轮|最后记录轮|",
            "|---|---:|---:|",
        ]
        for k, label in [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP50"),
            (KEY, "mAP50–95"),
        ]:
            lines.append(f"|{label}|{float(best[k]) * 100:.4f}%|{float(rows[-1][k]) * 100:.4f}%|")
        epochs = [int(x["epoch"]) for x in rows]
        lines += ["", f"轮次连续性：{epochs == list(range(epochs[0], epochs[-1] + 1))}。"]
        changes = [
            (a["epoch"], b["epoch"], float(a[KEY]) * 100, float(b[KEY]) * 100)
            for a, b in zip(rows, rows[1:])
            if float(b[KEY]) - float(a[KEY]) < -0.03
        ]
        resets = [
            (a["epoch"], b["epoch"])
            for a, b in zip(rows, rows[1:])
            if float(b.get("time", 0)) < float(a.get("time", 0))
        ]
        lines += [
            f"单轮mAP50–95下降超过3个百分点：{changes or '未检测到'}。",
            f"累计时间重置位置：{resets or '未检测到'}。这不是完整续训审计。",
        ]
    else:
        lines += ["", "暂无成绩，不能填入0当作实验结果。"]
    lines += ["", "## 阅读注意事项", ""]
    if name == "b2_direction4_bifpn_300e_seed1":
        lines += ["**旧异常记录：第44→45轮精度断崖，不纳入主对照；主对照使用同名加_rerun目录。**"]
    if name == "b0_yolo26n_baseline_300e_seed2":
        lines += ["**原未完成目录；正式结果使用b0_yolo26n_baseline_300e_seed2_rerun。**"]
    if name == "b2_direction4_bifpn_300e_seed2":
        lines += [
            "已核实第124轮恢复备份只将pretrained字段改为false，模型/优化器状态未改，不能据此认定从零训练；第152轮改workers4→0，应披露此差异。"
        ]
    if name == "fusion_resume_regression":
        lines += [
            "首次流程检查在保存第1轮后因测试断言未接受Path类型结束；后续带时间戳的恢复回归已通过。本目录不是正式训练结果。"
        ]
    if smoke:
        lines += ["小样本、短轮数只检查运行流程，低精度或零精度不用于判断方案效果。"]
    if snapshot:
        lines += ["这是上级实验某次恢复前的副本，不能作为额外种子或额外重复实验计数。"]
    if name.startswith(("f1_", "f2_")):
        lines += [
            "必须同时对照F0；仅超过另一个融合变体不足以证明有效。再与已有B0/B2比较精度和参数量，正收益需多种子确认。"
        ]
    lines += [
        "同seed但不同workers、恢复经历或训练时长，仍可能存在执行差异；不要只挑选最有利的参照。",
        "说明是可更新摘要；原始args.yaml、results.csv、权重与日志才是证据。",
        "",
        "## 文件怎么用",
        "",
        "- args.yaml：实际配置。results.csv：逐轮曲线和指标。",
        "- weights/best.pt：最佳权重，用于统一验证。weights/last.pt：最后权重；已完成实验可能已去除优化器，不能保证还能续训。",
        "- results.png及各类曲线/混淆矩阵：辅助分析，不能替代相同口径的数值比较。",
    ]
    (folder / "实验说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(folder), "title": title, "kind": kind, "status": status}


def write_evaluation_note(folder):
    folder = Path(folder)
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    protocol = protocol.get("settings", protocol)
    items = []
    for p in sorted(set(folder.glob("*_metrics.json")) | set(folder.glob("B[02]_seed*.json"))):
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "metrics" not in x:
            continue
        items.append((p, x))
    lines = [
        "# 独立验证结果说明",
        "",
        f"目录：`{folder}`",
        "",
        "目的：对已训练权重按固定设置重新计算指标，用于横向比较。这不是新训练、额外随机种子或独立训练重复。",
        "权重来源与哈希以结果JSON为准；详细类别指标保存在对应JSON。不要与训练CSV最佳值混用。",
        "",
        "|文件|权重来源|mAP50 %|mAP50–95 %|",
        "|---|---|---:|---:|",
    ]
    for p, x in items:
        m = x["metrics"]
        lines.append(
            f"|{p.name}|{fmt(x.get('checkpoint', '未记录'))}|{m.get('metrics/mAP50(B)', 0) * 100:.4f}|{m.get(KEY, 0) * 100:.4f}|"
        )
        side = [
            "# 单份验证结果说明",
            "",
            f"结果文件：{p.name}",
            f"权重：`{x.get('checkpoint', '未记录')}`",
            f"SHA256：`{x.get('sha256', '未记录')}`",
            "",
            "这是已训练权重的独立验证，不是新增训练。配置和整体对照见本目录《验证说明.md》。",
        ]
        p.with_suffix(".说明.md").write_text("\n".join(side), encoding="utf-8")
    if not items:
        lines += ["", "**未找到完整可解析的指标JSON，不当作验证成功。可能为历史失败或未完成的输出；原文件保留。**"]
    lines += [
        "",
        "## 实际评估设置",
        "",
        "```json",
        json.dumps(protocol, ensure_ascii=False, indent=2),
        "```",
        "",
        "mAP单位在表中为百分比，差值应使用“百分点”。该项目使用本地YOLO验证口径，不自动等同于官方VisDrone评测。",
    ]
    (folder / "验证说明.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "folder": str(folder),
        "title": "独立验证",
        "kind": "验证，不是训练",
        "status": f"{len(items)}份完整指标JSON",
    }


def attach_training_notes(model):
    def update(trainer, phase):
        try:
            write_training_note(trainer.save_dir, phase)
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            from ultralytics.utils import LOGGER

            LOGGER.warning(f"Experiment note could not be updated: {error}")

    model.add_callback("on_train_start", lambda trainer: update(trainer, "已启动"))
    model.add_callback("on_model_save", lambda trainer: update(trainer, "已保存检查点；训练可能继续"))
    model.add_callback("on_train_end", lambda trainer: update(trainer, "训练入口已结束"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    print(write_evaluation_note(args.run) if args.evaluation else write_training_note(args.run))

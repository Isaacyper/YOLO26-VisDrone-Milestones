# YOLO26 · VisDrone 研究里程碑

这是Isaacyper的YOLO26/VisDrone2019-DET研究源码归档，重点保存轻量化与精度改进过程中值得保留的阶段。
**默认分支包含F2源码和研究总览**。其他阶段通过下面的分支或标签独立查看、下载。

|阶段|内容|10类模型参数量|代码分支|为什么保留|
|---|---|---:|---|---|
|B2|方向四：原YOLO26上的残差加权BiFPN|2,604,742|[milestone/direction4-bifpn](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/tree/milestone/direction4-bifpn)|结构探索里程碑；为后续轻量融合提供经验，精度证据CSV单列|
|F0|方向六：FCM/MKP轻量化对照|570,992|[milestone/direction6-lightweight](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/tree/milestone/direction6-lightweight)|低参数、低延迟基线；不声称全指标超过原版|
|F2|F2：方向六＋单层残差BiFPN|605,812|[milestone/f2-residual-bifpn](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/tree/milestone/f2-residual-bifpn)|主要融合里程碑；三个种子验证及test-dev均值较F0提高，须同时报告速度和类别退化|
|D1|D1：F2＋两处DySample|612,020|[milestone/d1-dysample](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/tree/milestone/d1-dysample)|当前研究起点；验证均值提高，但小目标与test-dev收益不稳定，保留为研究候选|

## 实测结果：按对应对照比较

精度均为百分比，差值为百分点。以下表格来自已经完成的实验记录，没有为发布重新训练，也没有选择最高分种子替代三种子平均。

### F0 → F2：单层残差BiFPN融合里程碑

统一FP32，seed 0/1/2，640整图评价。

|划分|F0平均mAP50–95|F2平均mAP50–95|平均差值|
|---|---:|---:|---:|
|验证集|17.4415%|17.8191%|+0.3776|
|test-dev|14.0477%|14.2475%|+0.1997|

F2在当前RTX4060 Laptop、FP32条件下，batch1完整推理链的中位延迟比F0增加约8.05%（10.064→10.874 ms）。
带篷三轮车等类别仍有退化；因此这是需要权衡速度和类别表现的精度候选，并非全指标无代价提升。

### D0 → D1：DySample研究版本

D0是F2在DySample工程内重新训练的同工程对照；它不是上表中那三份历史F2权重。两组结果不可混用。

|划分|D0平均mAP50–95|D1平均mAP50–95|平均差值|
|---|---:|---:|---:|
|验证集|17.8968%|18.1483%|+0.2515|
|test-dev|14.3268%|14.4209%|+0.0941|

D1小目标专项AP的三对差值为+0.7541/−0.0648/−0.2560个百分点；test-dev三对差值为+0.3877/−0.0272/−0.0784。
D1的batch1完整推理链中位延迟比D0约+6.54%。因此保留D1作为研究候选与后续研究起点，当前证据不支持称为稳定小目标提升或最佳部署版本。

### B0 → B2：早期方向四结构探索

此表只使用各次训练CSV中的最佳mAP50–95行，**不是上面FP32口径**，不能横跨两种表格排序。

|seed|B0 CSV最佳mAP50–95|B2 CSV最佳mAP50–95|差值|
|---|---:|---:|---:|
|0|18.2630%|18.5300%|+0.2670|
|1|18.2480%|18.5190%|+0.2710|
|2|18.0510%|18.1880%|+0.1370|

B0 seed0早停于268轮，其他所列运行300轮；早期workers、恢复历史存在差异。
B0 seed2及B2 seed1使用记录中的重跑目录，异常旧目录不重复计数。
B2作为BiFPN结构探索和后续融合的来源保留，不把这张CSV表视为严格同执行流程的FP32确证。

## 版本和使用

- v1-direction4-bifpn：[B2源码ZIP](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/archive/refs/tags/v1-direction4-bifpn.zip)。
- v1-direction6-f0：[F0源码ZIP](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/archive/refs/tags/v1-direction6-f0.zip)。
- v1-f2：[F2源码ZIP](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/archive/refs/tags/v1-f2.zip)。
- v1-d1：[D1源码ZIP](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/archive/refs/tags/v1-d1.zip)。

每个源码分支包括原Ultralytics代码、模型配置、训练/验证脚本、来源SHA256清单和结果摘要。保留同期对照配置有助于理解消融实验。
源码复制后逐文件哈希一致；四个10类模型均已在CPU构建并完成有限值推理检查，见[PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json)。

默认F2模型的无数据构建检查：

    python -c "from ultralytics.nn.tasks import DetectionModel; m=DetectionModel('configs/models/yolo26n_direction46_single.yaml', nc=10, verbose=False); print(sum(p.numel() for p in m.parameters()))"

原训练入口保存了本地研究路径和溯源检查。迁移机器需要准备VisDrone数据、原预训练权重及对应manifest/论文资料，并配置数据路径；本仓库是原样研究归档，尚不是跨机一键训练发行包。
数据、训练权重、预测、日志、凭据、环境私密文件和论文PDF不上传；这些仍需在本地单独保留。

## 证据边界

- 本地转换标注的Ultralytics指标，不是官方VisDrone挑战成绩。
- test-dev曾被历史研究查看过，不再视为全新盲测；后续改进不能持续按test-dev调参再声称独立盲测。
- 三个种子的平均提升不能单独证明统计显著。
- 不同工程、不同种子、CSV与FP32的结果按组分别报告，不比较最高分种子后宣称全面领先。
- 源码是当前本地阶段归档，可能含后来完善的训练恢复工具；发布过程未重设计模型或新增完整训练。
- 未把Wise-IoU、MGD诊断迭代、ShapeIoU、关闭NWD、TaskAlignedFocal、普通卷积替换等其他候选源码批量放入成功版本。
- 正在进行的C1原图裁剪实验尚无最终结论，本次不列入成功里程碑。

## 来源与许可

Ultralytics及原研究代码保留[AGPL-3.0许可](LICENSE)和源文件版权声明。
这些是论文思想在本项目中的研究适配，不等同于全部原论文的官方逐项复现。

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [EfficientDet / BiFPN](https://arxiv.org/abs/1911.09070)
- 方向六结构与NWD思想参考FBRT-YOLO、SOD-YOLO，具体实现以各分支源码注释为准。
- [DySample论文](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html) / [作者实现](https://github.com/tiny-smart/dysample)

原始分组数值见[results/summary.json](results/summary.json)，版本映射见[milestones.json](milestones.json)。

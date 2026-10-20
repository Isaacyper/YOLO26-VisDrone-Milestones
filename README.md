# 方向四：原YOLO26上的残差加权BiFPN

本分支：milestone/direction4-bifpn。实验编号：**B2**。

结构探索里程碑；为后续轻量融合提供经验，精度证据CSV单列。

模型配置：[configs/models/yolo26n_direction4.yaml](configs/models/yolo26n_direction4.yaml)。
本分支按原工程保存源码，模型配置和核心实现不在发布时重新设计；同期对照配置也可能保留在configs中。
逐文件SHA256见[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，精度表及适用限制见[results/summary.json](results/summary.json)。

## 运行边界

这是历史实验源码归档，不包含数据集、预训练/训练权重、原始日志、预测和论文PDF副本。
原脚本使用本地数据路径；迁移到其他机器时需要修改configs/visdrone2019-det.yaml，并准备对应训练数据、原YOLO26n预训练权重和脚本引用的本地溯源资料。只克隆源码不能恢复已有训练模型。
部分verify/provenance脚本依赖原研究目录的manifest、论文和历史对照，这些资料不在公开仓库内，因此不声明所有原脚本可跨机一键运行。不要删除校验后声称完整复现。

在已安装兼容PyTorch的Python环境中，可先从本分支目录进行不依赖数据集的模型构建检查：

    python -c "from ultralytics.nn.tasks import DetectionModel; m=DetectionModel('configs/models/yolo26n_direction4.yaml', nc=10, verbose=False); print(sum(p.numel() for p in m.parameters()))"

原实验通常采用300轮上限、640输入、batch8、SGD、AMP、原YOLO26n预训练。方向四早期workers和恢复经历与后期不同；后期workers0。具体复现实验必须以对应原args/配置和结果说明为准，不能把更换模型但沿用不同损失的运行当作同一个对照。
本版本约定NWD混合权重为0；方向四/原版对照与方向六混合版的损失处理不同。

## 结论与归属

结果摘要以组内对照解释，不将CSV和独立FP32结果混用，也不按不同工程的最好种子排序。
这些是基于Ultralytics的研究适配，不是所有参考论文的逐项官方复现。

- Ultralytics源码沿用原作者AGPL-3.0许可，见[LICENSE](LICENSE)及源码文件头。
- BiFPN思想参考EfficientDet；FCM/MKP/NWD参考方向六的FBRT-YOLO、SOD-YOLO研究。
- DySample参考[Learning to Upsample by Learning to Sample, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html)及[作者实现](https://github.com/tiny-smart/dysample)。
- [研究总览](https://github.com/Isaacyper/YOLO26-VisDrone-Milestones/tree/publish/milestones)区分已验证收益与研究候选。

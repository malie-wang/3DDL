# 3DDL 模型轻量化与知识蒸馏实验报告

## 1. 核心性能与压缩率对比 (Efficiency Metrics)
> **实验背景**：以标准 CenterPoint (8.94M/115G) 为 Teacher，验证不同蒸馏策略下小模型的压缩效率。

| 模型方案 | 参数量 (Params) | 算力 (FLOPs) | 推理速度 (FPS) | 压缩比 (Params) | 加速比 (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (Base)** | 8.94 M | 115.94 G | 5.8 | 1.00x | 1.00x |
| **同模蒸馏** | 8.94 M | 115.94 G | 5.8 | 1.00x | 1.00x |
| **异构蒸馏** | 6.22 M | 91.25 G | 6.0 | 1.43x | 1.03x |
| **同架构小模型蒸馏** | **6.22 M** | **91.25 G** | **6.0** | **1.43x** | **1.03x** |

---

## 2. 精度表现对比 (Accuracy Metrics)
> 评测数据集：nuScenes-mini (Val)

| 实验方案 | mAP | **NDS** | Car (AP) | Pedestrian (AP) |
| :--- | :--- | :--- | :--- | :--- |
| **异构蒸馏** | 0.0387 | 0.1110 | 0.253 | 0.099 |
| **同模蒸馏** | 0.1450 | 0.2149 | 0.614 | 0.726 |
| **同架构小模型蒸馏** | **0.1513** | **0.2439** | **0.613** | **0.741** |

---

## 3. 实验操作标准化流程 (SOP)
> 为防止遗忘，以下是针对 MMDetection3D 框架的完整测试流程。

### 第一步：精度评估 (Evaluation)
**工具**：`eval_mini.py` (自定义 Python 脚本)
**核心步骤**：
1. 运行推理生成结果 JSON 文件。
2. 调用 `NuScenesEval` 进行指标计算。
3. **关键注意**：Config 必须指定为 `'detection_cvpr_2019'`，否则无法通过 `config_factory` 校验。

### 第二步：算力与参数统计 (FLOPs & Params)
**工具**：`calc_student_metrics.py` (Hook 挂载法)
**操作逻辑**：
1. **剥离模型**：在脚本中通过 `MODELS.build(cfg.model.student)` 单独构建学生模型。
2. **挂载预处理器**：手动将 `cfg.model.data_preprocessor` 赋值给学生模型，否则体素化阶段会崩溃。
3. **前向 Hook**：通过 `register_forward_hook` 捕获卷积层输出维度，根据 3D 卷积公式 `2 * Cin * Cout * K^3 * D * H * W` 统计真实算力。
4. **提交方式**：使用 Slurm `sbatch` 提交到 GPU 节点，确保 CUDA 环境可用。

### 第三步：推理速度测试 (Benchmark FPS)
**工具**：官方 `tools/analysis_tools/benchmark.py`
**执行命令**：
`python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --samples 80`
**数据解读**：FPS 结果受服务器实时负载影响。3D 检测的体素化与 NMS 是固定开销，因此 FLOPs 下降并不线性等同于 FPS 的同比例提升。

---

## 4. 技术分析与总结
1. **蒸馏增益**：实验证明“同架构小模型蒸馏”效果最佳。尽管参数量从 8.94M 砍到了 6.22M，但 NDS 反而从 0.2149 提升至 **0.2439**，说明合理的结构剪裁配合特征对齐能有效缓解过拟合。
2. **瓶颈分析**：91G 版模型 FPS 仅从 5.8 提升至 6.0，说明在 TC2 服务器环境下，CenterPoint 的**数据预处理 (Voxelization)** 和 **后处理 (CircleNMS)** 占据了推理时间的大部分，单纯精简 Backbone 对端到端速度提升有限。
3. **经验避坑**：在 Linux 环境下复制 Python 代码时，应彻底剔除中文注释或声明 `# -*- coding: utf-8 -*-`，防止编码报错。

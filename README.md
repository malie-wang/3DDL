# 3DDL 模型轻量化与知识蒸馏实验报告

## 1. 核心性能与效率对比 (Efficiency Metrics)
> **实验背景**：对比 Teacher/Base 与不同 Student 策略下的算力开销与推理速度。

| 模型方案 | 参数量 (Params) | 算力 (FLOPs) | 推理速度 (FPS) | 压缩比 (Params) | 加速比 (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (Base) / 同模** | 8.94 M | 115.94 G | 5.8 sample/s | 1.00x | 1.00x |
| **Student (同架构小模型)** | **6.22 M** | **91.25 G** | **6.0 sample/s** | **1.43x** | **1.03x** |

---

## 2. 整体精度指标对比 (Overall Metrics)
> 重点评估不同蒸馏方案对核心指标 (NDS, mAP) 及各项误差 (mATE, mASE 等) 的影响。

| 实验方案 | mAP | **NDS** | mATE | mASE | mAOE | mAVE | mAAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **异构蒸馏** | 0.0387 | 0.1110 | 0.7796 | 0.6674 | 1.0163 | 1.0591 | 0.6364 |
| **同模蒸馏 (115G)** | 0.1450 | 0.2149 | 0.5741 | 0.5272 | 1.0117 | 1.0981 | 0.4752 |
| **同架构小模型 (91G)** | **0.1513** | **0.2439** | **0.5826** | **0.5881** | **1.1213** | **0.6461** | **0.5011** |

---

## 3. 类别精度细分对比 (Per-class AP)
> 已严格剔除测试集中无真实样本 (GT=0) 的无效类别 (construction_vehicle, trailer, barrier)。

| 类别 (Object Class) | 样本数 | 异构蒸馏 (AP) | 同模蒸馏 (AP) | 同架构蒸馏 (AP) |
| :--- | :--- | :--- | :--- | :--- |
| **Car (汽车)** | 2568 | 0.253 | 0.614 | **0.613** |
| **Truck (卡车)** | 124 | 0.029 | 0.060 | **0.106** |
| **Pedestrian (行人)** | 1358 | 0.099 | 0.726 | **0.741** |
| **Motorcycle (摩托)** | 259 | 0.000 | 0.039 | **0.054** |
| **Bus (公交)** | 41 | 0.005 | **0.012** | 0.000 |
| **Bicycle (自行车)** | 52 | 0.000 | 0.000 | 0.000 |
| **Traffic Cone (锥桶)**| 39 | 0.000 | 0.000 | 0.000 |

---

## 4. 实验操作标准化流程 (SOP)
> 针对 MMDetection3D 框架的测试流程记录。

### 第一步：精度评估 (Evaluation)
**工具**：`eval_mini.py` (自定义脚本)
**核心步骤**：
1. 运行推理生成 `results_nusc.json`。
2. 调用 `NuScenesEval` 进行指标计算。
3. **关键注意**：Config 必须指定为 `'detection_cvpr_2019'`。

### 第二步：算力与参数统计 (FLOPs & Params)
**工具**：`calc_student_metrics.py` (Hook 挂载法)
**操作逻辑**：
1. **剥离模型**：在脚本中通过 `MODELS.build(cfg.model.student)` 单独构建学生模型。
2. **挂载预处理器**：手动将 `cfg.model.data_preprocessor` 赋值给学生模型，确保体素化正常运行。
3. **前向 Hook**：通过 `register_forward_hook` 捕获输出维度，利用 3D 卷积公式统计真实算力。
4. **提交测试**：使用 `sbatch` 提交到 GPU 节点 (TC2)。

### 第三步：推理速度测试 (Benchmark FPS)
**工具**：官方 `tools/analysis_tools/benchmark.py`
**执行命令**：
`python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --samples 80`
**结论支撑**：FLOPs 下降 (115G -> 91G) 未能带来 FPS 的同比例提升，证实 3D 检测中数据预处理 (Voxelization) 与后处理 (CircleNMS) 占据了固定的推理时间开销。

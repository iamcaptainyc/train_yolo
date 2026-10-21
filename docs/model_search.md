# 四模型 Optuna 自动搜索

`search_yolo.py` 按轮串行训练：**yolov8n → yolov8s → yolo11n → yolo11s → 下一轮**。每个模型有独立的 Optuna TPE study，最大化该候选 `best.pt` 在验证集上的 mAP50-95。最终在两个 n 架构间选一个、两个 s 架构间选一个，导出 `best_n.pt` 和 `best_s.pt`。结果是本次搜索预算内的最佳候选，不保证全局最优。

默认每个候选训练完都会重新评估其 `best.pt` 的验证集和测试集，独立保存结果；**测试集只记录，Optuna、排名和最终选择都不读取测试指标**。如果人为根据每轮测试分数修改搜索，测试集也会产生选择偏差。正式实验建议 `--test-policy final`，仅在搜索结束后评估最终两个模型；默认 `every-trial` 则满足逐候选自动测试的使用方式。

## 安装和使用

在仓库目录执行，使用与你之前训练相同的数据 YAML；必须含独立的 train/val/test：

```bash
git pull --ff-only
python -m pip install -r requirements-search.txt

# 只查看计划，不启动训练、不下载权重、不创建搜索目录
python search_yolo.py --data yolo_datasets/animals-detection-images-dataset/data.yaml --dry-run

# 建议先使用独立目录跑通流程：每个模型一次，每次最多 3 epoch
python search_yolo.py \
  --data yolo_datasets/animals-detection-images-dataset/data.yaml \
  --output runs/search_smoke --trials 1 --epochs 3 --patience 2 \
  --batch 32 --eval-batch 16 --workers 2 --device 0

# 正式搜索：5 轮、4 个模型，共 20 次训练，每次最多 100 epoch
python search_yolo.py \
  --data yolo_datasets/animals-detection-images-dataset/data.yaml \
  --output runs/animals_search --trials 5 --epochs 100 --patience 20 \
  --batch 64 --eval-batch 32 --workers 4 --device 0
```

96GB 显存和 110 CPU 核并不保证容器 RAM 足够，因此默认固定 batch=64、workers=4，并关闭图片缓存。先检查资源再调高，不会自动尝试 batch=256/512 或 32 workers。`--eval-batch` 独立于训练 batch，避免测试阶段使用过大的 batch。脚本单卡串行执行，不会同时启动四个训练任务。

初始权重优先从 `--weights-dir`（默认 `weights/`）复制；缺失则使用 Ultralytics 官方下载器获取。首次保存到搜索目录的 `initial_weights/`，记录 SHA-256。每个新候选从这份固定的预训练权重开始，**不会从上一候选的 best.pt 接着微调**。只有同一中断候选才用自己的 last.pt 恢复。

四模型轮转、训练、评估均自动完成，不需要分别调用 `train.py`。每个 GPU 阶段在独立子进程中，Linux 复用 `training_guard.py`：阶段结束后清理它自己的进程组，再进入下一阶段。守护本身被 SIGKILL 或容器整体终止仍无法保证清理；Windows 不提供 Linux 进程组的同等回收能力。

## 搜索参数

每个模型第一个候选使用固定 AdamW 基线，后续由 Optuna 建议：

| 参数 | 范围 | 首个候选 |
| --- | --- | --- |
| optimizer | AdamW / SGD | AdamW |
| lr0 | 0.0001～0.01，对数采样 | 0.001 |
| lrf | 0.01～0.2，对数采样 | 0.01 |
| momentum | 0.85～0.98 | 0.937 |
| weight_decay | 0.00001～0.001，对数采样 | 0.0005 |
| hsv_s | 0.3～0.8 | 0.7 |
| hsv_v | 0.1～0.5 | 0.4 |
| scale | 0.2～0.7 | 0.5 |
| mosaic | 0.3～1.0 | 1.0 |

显式使用优化器，避免 `optimizer=auto` 忽略建议的 lr0。训练轮数、batch、分辨率保持固定，保证候选比较条件一致。候选训练有 Ultralytics `patience` 早停；当前不使用 Optuna 的逐 epoch pruning。TPE 前三个候选是启动采样，之后参考已完成候选；5 个候选只是初步搜索，预算充足可以增加 `--trials`。

训练以 Ultralytics 保存的 best.pt 为候选，然后显式重新评估验证集。Optuna 优化的是这个 checkpoint 的验证 mAP50-95，而非最后一个 epoch，也不一定是 results.csv 中单项 mAP 最高的那个 epoch（checkpoint 选择由所装 Ultralytics 的规则决定）。

评估固定使用 conf=0.001、IoU=0.7、max_det=300，保存 Precision、Recall、mAP50、mAP50-95、各类别指标，以及推理阶段每图片毫秒数。测速仅作参考，受 batch/硬件/系统负载影响，不作为选优目标。

## 输出

```text
runs/animals_search/
  run.json                         # 搜索配置和数据指纹，恢复时检查一致性
  optuna.sqlite3                   # 四个独立 study 的持久化记录
  iterations.csv                   # 所有候选的参数、验证/测试指标和状态
  summary.json                     # 详细结果及当前获胜模型
  initial_weights/                 # 固定的初始权重、校验值、下载日志
  rounds/round_0001.json            # 本轮四模型结果快照，逐阶段更新
  models/yolov8n/trial_0000/
    config.json                    # 固定训练条件和建议参数
    status.json
    train_job.json / val_job.json / test_job.json
    train_result.json / val_result.json / test_result.json
    train.log / val.log / test.log  # 每阶段日志，中断重试追加
    training/weights/best.pt
    training/weights/last.pt
    training/results.csv           # Ultralytics 原始逐 epoch 记录和图表
    evaluation/val/...
    evaluation/test/...
  best/
    best_n.pt
    best_s.pt
    selection.json                 # 获胜来源、参数、验证和测试指标
```

每个候选的原始权重、日志及指标保留。`best/` 随搜索更新，进程成功结束后才代表此次预算下的最终结果。若部分候选失败，脚本仍可能已经导出可用的最优权重，但会以非零状态提示检查失败记录。

## 中断、失败和扩展

重跑**同一命令、同一输出目录**即可继续：

- 已完成的训练、验证、测试阶段按记录复用；缺失或被修改的完成权重会报错，不拿旧分数对应新权重。
- 未完成候选保留 RUNNING；有可恢复的 last.pt 时继续同一训练。没有 checkpoint 时从初始权重重试该候选，保留旧的未完成目录。只能恢复到已保存 epoch，不能恢复未保存的 batch。
- 测试失败不撤销有效的验证分数、不重跑已完成训练；下次只重试失败的评估。
- SIGKILL、CUDA/主机内存不足、明显本地存储错误会停止整批，避免反复启动失败进程。处理资源问题后重跑；不自动杀其他训练进程、不自动降低 batch 改变实验。
- 其他阶段异常记为失败候选，保留日志并继续后续模型。失败候选计入 `--trials` 总尝试预算；想增加有效搜索量，可提高目标试验数。
- 同一输出目录有文件锁，只允许一个搜索协调器，避免重复使用 GPU 或破坏 SQLite。

`--trials 5` 表示**每个模型累计目标五次尝试**，恢复时不会再额外执行五次；增加到 `--trials 10` 才会补充到十次。

允许恢复时调整 trials、device、workers、eval-batch；已完成阶段保持原结果。更改 epochs、batch、imgsz、seed、patience、AMP、数据 YAML、初始权重目录或测试策略，需要使用新的输出目录。继续同一中断训练时，资源参数是否可覆盖仍由所安装的 Ultralytics 版本决定，实际以该阶段日志为准。需要公平测速时保持 eval-batch 一致。

数据指纹包含 YAML 和转换器 report.json（如存在）的 SHA-256，没有逐张重新哈希数据集；搜索期间保持图片、标签和划分不变。初始权重固定为复制到搜索目录的版本，外部权重目录后来修改不会改变它。每阶段记录 Python、PyTorch 和 Ultralytics 版本，恢复时应尽量保持环境一致。建议参数随机种子按模型和候选编号派生，避免每次重启重复随机启动序列；浮点计算仍不保证跨硬件或库版本完全一致。

## 使用选出的模型

```bash
python train.py --mode predict --model runs/animals_search/best/best_n.pt --source /你的新图片目录 --device 0 --name best_n_predict
python train.py --mode predict --model runs/animals_search/best/best_s.pt --source /你的新图片目录 --device 0 --name best_s_predict
```

本地验证使用真实 Optuna/SQLite 和模拟 GPU 进程边界，检查轮转、恢复、测试失败重试、配置隔离和验证集选优；没有替你在 AutoDL 上执行实际搜索或 GPU 训练。正式长搜索前先跑独立的 smoke 目录。

参考：[Optuna 持久化 study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html)、[Ultralytics 训练](https://docs.ultralytics.com/modes/train/)、[评估](https://docs.ultralytics.com/modes/val/)。

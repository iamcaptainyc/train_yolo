# 四模型自动调参

使用短命令 `tune.py`（原 `search_yolo.py` 仍可用），依次训练 YOLOv8n、YOLOv8s、YOLO11n、YOLO11s。四个模型各有独立 Optuna study，以 **val mAP50-95** 选出本次预算内最好的 n 和 s。测试集不参与调参或选优。

## 直接运行

```bash
git pull --ff-only
python -m pip install -r requirements-search.txt

# 查看计划，不训练、不下载、不创建输出目录
python tune.py --data yolo_datasets/camping_all_classes/data.yaml --dry-run

# 先比较四个架构的默认优化器基线：共 4 次训练
python tune.py --data yolo_datasets/camping_all_classes/data.yaml --trials 1 --workers 16

# 默认每模型 8 次，包含基线；共 32 次训练
# 若之前已完成上述基线，会接续同一实验，补足到 8 次
python tune.py --data yolo_datasets/camping_all_classes/data.yaml --workers 16
```

不需要自己命名：自动目录例如 `runs/tune/camping-all-classes-a1b2c3d4e5/`，模型/候选使用 `v8n/t000`、`v8s/t000`、`v11n/t001`、`v11s/t001`。实际目录会打印出来，示例末尾不是固定值。

名称由数据 YAML、可用的转换报告和实验条件生成短指纹。相同命令自动续跑；改 epochs、分辨率、seed、batch 设置等会生成另一个目录。只增加 `--trials` 或修改 workers/device/eval-batch 不改变目录。也可显式指定 `--output`。旧版搜索空间不能接续到新版；新版自动目录会隔离旧结果。

指纹只读取 YAML 和同目录 report.json，不逐张校验图片。数据内容若原地修改而这些文件不变，无法自动识别；此时更换输出目录。实验期间保持数据不变。

## 策略与预算

默认每模型 8 次候选：

1. 第 1 次：batch=64、optimizer=auto，不覆盖学习率、权重衰减和数据增强参数，建立官方默认优化器基线。
2. 第 2～4 次：固定 AdamW、lr0=0.001、weight_decay=0.0005、mosaic=1.0，分别用 batch=64/128/256，先比较 batch 的影响。
3. 第 5～8 次：TPE 根据该模型已有验证结果，集中搜索下表四个参数。

| 参数 | 搜索范围 | 考虑 |
| --- | --- | --- |
| batch | 64 / 128 / 256 | 保留指定的档位；影响吞吐、显存和优化过程 |
| lr0 | 0.0001～0.003，对数采样 | 围绕 AdamW 预训练微调的 0.001 附近搜索，避免小预算横跨过大的范围 |
| weight_decay | 0.00001～0.001，对数采样 | 覆盖较弱到较强正则化 |
| mosaic | 0～1 | 相机陷阱背景固定，比较减少拼接增强是否更适合该数据 |

除基线外，优化器固定 AdamW，避免把 SGD 和 AdamW 的不同学习率需求混在一个小预算里；显式指定优化器也避免 auto 忽略建议的 lr0。其余学习率调度、颜色增强、损失权重保留安装版本的默认值。所有候选使用相同的 epochs、imgsz、seed、patience 等控制设置；沿用脚本的 close_mosaic=min(10, epochs//5)、cache=False、deterministic=True，因此基线指默认优化器/超参数基线，并非完全无覆盖的原生 CLI 命令。

`--batch 128` 会关闭 batch 搜索：第 1 次 auto 基线，第 2 次固定 AdamW 参考点，后续由 TPE 搜索三个连续参数（TPE 启动期仍有随机采样）。指定 `--trials 1` 只跑基线，不执行排队的后续候选。trial 数是每模型累计尝试数，失败也计入预算。

默认 8 次只是初步搜索，并不能证明参数最优。可增加到 `--trials 12` 或 20 接续搜索。原先 5 次搜索约 10 个维度，信息太少；新版先做可解释的基线和 batch 比较，再做窄范围搜索。调参不保证超过基线，基线也参与最终选优。

每次最多 100 epoch，patience=20；支持修改 `--epochs`、`--patience`。使用 Ultralytics 早停，不启用 Optuna 逐 epoch 剪枝，避免把早期指标波动误判为无效实验。各候选从固定的初始预训练权重重新开始；只有中断的同一候选从自己的 last.pt 恢复。

## 评估和结果

默认 `--test-policy final`：每个候选训练完成后重新验证 best.pt，全部候选结束后仅测试选中的 n、s，共两次测试。若要保留原来的逐候选测试方式，传 `--test-policy every-trial`；测试指标仍不参与选优。扩展预算后会评估新的获胜者，不应据测试分数调整搜索。

每阶段结果立即保存，包含：

```text
runs/tune/<自动名称>/
  run.json                     # 实验身份
  optuna.sqlite3               # 四个 study
  REPORT.md                    # 人类可读：按模型/尺寸分组、基线变化和最终指标
  iterations.csv               # 每个候选的传入参数、指标和状态
  summary.json                 # 详细结果和获胜来源
  rounds/round_0001.json        # 每轮结果
  initial_weights/             # 固定的初始权重及校验值
  v8n/t000/                    # 另外三个模型结构相同
    config.json                # 传入参数；auto 基线不虚构 lr0
    status.json
    train.log / val.log / test.log
    train_result.json / val_result.json / test_result.json
    training/args.yaml         # Ultralytics 保存的配置
    training/results.csv       # 逐 epoch 日志（含学习率）
    training/weights/best.pt
    training/weights/last.pt
    evaluation/val/
    evaluation/test/
  best/best_n.pt
  best/best_s.pt
  best/selection.json
```

auto 的实际优化器和初始学习率看 train.log；args.yaml 可能仍记录 auto 配置。iterations.csv 的空白参数表示未显式覆盖，不是零。每个候选保留完整结果、权重，不自动删除旧实验。

评估固定 conf=0.001、IoU=0.7、max_det=300，记录 Precision、Recall、mAP50、mAP50-95、逐类指标和推理毫秒数。推理时间受评估 batch 和系统负载影响，不作为选优指标。best.pt 的保存由 Ultralytics 的规则决定，脚本再对该 checkpoint 的验证 mAP50-95 排名。

## 中断与资源

Ctrl+C 后重跑同一命令即可：完成阶段复用；未完成训练有 last.pt 时恢复到已保存 epoch，无 checkpoint 时重试该候选。未保存的 batch 无法恢复。测试失败下次只重试测试，不重训。SQLite 和输出锁防止两个协调器混写同一实验。

CUDA/主机内存不足、SIGKILL、磁盘异常会停止整批并保留恢复信息，不自动降低 batch 改变实验。其他阶段错误保留日志，可能继续后续候选，最终以非零退出码提示失败。每个阶段使用独立子进程，Linux 使用 training_guard 清理自身进程组；整个容器被强制终止仍不能保证清理。

默认 workers=4、eval-batch=32、device=0。96GB GPU 可尝试 workers=16，但需看容器 RAM 和磁盘吞吐，不按 110 CPU 核直接开满 workers。图片缓存关闭。训练串行运行，每次只启动一个模型。权重优先读取 weights/，缺失则由 Ultralytics 下载。

## 有没有公认的最佳参数？

没有适用于所有数据的最佳组合。官方默认值是合理起点；相机陷阱数据的分布、类别不均衡、小目标比例和训练规模都可能改变最适合的学习率及增强强度。先看基线；有预算再调少量关键参数。相比微调很多参数，正确标注、合理类别、无泄漏划分更优先。

参考：[Ultralytics 训练参数](https://docs.ultralytics.com/modes/train/)、[微调指南](https://docs.ultralytics.com/guides/finetuning-guide/)、[超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning/)。本地回归测试使用真实 Optuna/SQLite 和模拟 GPU，不代表已在远端数据上测出精度或速度提升。

## 阅读汇总报告

优先打开实验目录下的 `REPORT.md`：先展示最佳 n/s 及权重链接，分别列出验证集和测试集的 mAP50-95、mAP50、Precision、Recall；再按 n/s 比较架构，按四个模型列出全部候选、参数与相对各自基线的提升。指标统一以百分数显示，提升用百分点表示。未评估显示“—”，失败单独提示，不会用 0 代替缺失结果。

报告逐阶段更新。只有所有要求的阶段成功结束才标记“本次搜索完成”，中断时可查看阶段性结果。`auto` 标记的是配置，实际选中的优化器和学习率见候选训练日志。已有本版本搜索只需重跑同一命令，即可利用完成记录生成报告，无需重新训练；更早版本的搜索空间仍不能混用。

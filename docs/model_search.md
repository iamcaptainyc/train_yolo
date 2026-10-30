# 四模型自动调参

使用短命令 `tune.py`（原 `search_yolo.py` 仍可用），依次训练 YOLOv8n、YOLOv8s、YOLO11n、YOLO11s。四个模型各有独立 Optuna study，以 **val mAP50-95** 选出本次预算内最好的 n 和 s。测试集不参与调参或选优。

## 三种训练模式

`--mode standard` 保留原有行为（默认），旧命令和旧实验仍可恢复。新增两种纯 Optuna 模式，无需大模型/API：

| 模式 | 训练过程 | 默认参数 |
| --- | --- | --- |
| `standard` | 所有候选同尺寸训练，只有普通早停，不使用 Pruner | 640，最多 100 epoch；搜索 batch；逐候选 test |
| `progressive` | 小尺寸搜索 → 晋级配置从官方权重重新进行大尺寸训练 | 448/80 epoch → 640/250 epoch；batch 固定 64；每个 COMPLETE 候选立即 test |
| `prune` | 始终使用正式尺寸，30/80 轮筛选，幸存者继续完整训练 | 640，最多 250 epoch；batch 固定 64；每个 COMPLETE 候选立即 test |

两个新模式每个模型独立使用 **SuccessiveHalvingPruner**，默认检查点 `--prune-epochs 30 80`。每轮向 Optuna 上报真实 epoch 的 val mAP50-95；只有检查点才决定是否淘汰。内部将检查点映射为 successive-halving 的资源阶梯，约保留已到达同一阶梯候选中较好的半数；这是异步筛选，不保证固定晋级人数。不会在 60、120 轮偷偷增加检查点。基线也记录指标作为比较依据，但免剪枝。

同一阶段从开始就使用完整轮数的学习率计划：30→80→250 表示中间检查，不是反复修改 epochs 或重启优化器。通过最后检查点后不再剪枝，使用普通 patience 早停；新模式在最后有效检查点之前暂缓普通早停，防止绕过筛选。小尺寸筛选阶段关闭普通早停，最多运行 search-epochs 轮。Pruner 提前终止候选不属于网络通道剪枝。

### 模式一：小尺寸筛选，再大尺寸训练

```bash
python tune.py \
  --mode progressive \
  --data yolo_datasets/animals_30k_v4/data.yaml \
  --models yolo11n.pt yolo11s.pt \
  --trials 12 --search-imgsz 448 --search-epochs 80 \
  --imgsz 640 --epochs 250 --prune-epochs 30 80 \
  --promote 3 --batch 128 --eval-batch 32 --workers 16
```

每模型搜索次数包含基线。先完成各模型的小尺寸筛选；80 轮是该阶段终点，因此小尺寸阶段只在第 30 轮剪枝。每模型从完成筛选的非基线候选中按验证分数最多选择 3 组，再额外带上官方基线。候选不足时按实际数量晋级。正式阶段在 30/80 轮剪枝，最多训练 250 轮。

**只继承参数，不继承小尺寸权重**；正式阶段重新从同一官方预训练权重开始，正式阶段内的晋级不重新初始化。全部 train 图片都参与训练，不抽小子集。训练内验证、训练后的 val 和最终 test 始终使用正式 `--imgsz` 和 `--eval-batch`，避免不同验证尺寸干扰比较；因此小尺寸阶段仍有正式尺寸验证成本。

筛选和正式训练使用独立 SQLite/Study：`screen/REPORT.md` 只供筛选，根目录 `REPORT.md` 才是正式结果。`promotion.json` 记录来源候选、参数及晋级分数。默认筛选和正式阶段的每个 COMPLETE 候选都立即运行 test，但测试指标不参与晋级或最终选优。正式阶段同样允许 Pruner 淘汰参数迁移后表现不好的候选。

### 模式二：始终使用正式尺寸，分阶段剪枝

```bash
python tune.py \
  --mode prune \
  --data yolo_datasets/animals_30k_v4/data.yaml \
  --models yolo11n.pt yolo11s.pt \
  --trials 12 --imgsz 640 --epochs 250 \
  --prune-epochs 30 80 --batch 128 --eval-batch 32 --workers 16
```

所有候选始终按 640 训练：前 30 轮初筛 → 至 80 轮复筛 → 幸存者继续至 250 轮上限或早停。**三个阶段是同一次训练的连续预算，不会重新训练前 30/80 轮。** 默认每个 COMPLETE 候选立即测试并更新报告，再启动下一个候选；已剪枝候选不会测试。显式 `--test-policy final` 才会恢复仅测试最终最佳 n/s 的行为。

两种模式均无需手动命名，支持 `--dry-run` 预览；只训练一个模型就只传一个模型名。两个新模式未指定 `--batch` 时固定为 64；`--batch 128` 固定为 128。本次保留现有 lr0/weight_decay/mosaic 搜索空间，没有引入额外参数或修改网络结构。

### 恢复与报告

旧实验可从 `final` 升级为 `every-trial`，不会重训或新建 Optuna trial。其他配置不变时，自动目录会优先恢复对应旧实验（新策略目录尚不存在时）；也可显式指定原 `--output`。恢复时先补齐已 COMPLETE 候选的 test，再继续训练。若旧命令显式写了 `--test-policy final`，删除该参数或改为 `every-trial`。

- Ctrl+C 或资源故障后重跑相同命令；已有 train/val/test 结果复用，训练从自己的 last.pt 恢复，不重置学习率计划。
- `PRUNED` 是正常淘汰，保留最后权重、逐轮 `epochs.jsonl` 和 `pruned.json`，报告显示淘汰轮数及原因；恢复时不会重新训练。
- 根目录 `iterations.csv` 增加阶段、训练/验证尺寸、淘汰轮数、淘汰原因、来源候选字段。报告按模型/size 展示最终指标，并链接筛选报告和晋级名单。
- 模式、检查点、尺寸等参与实验指纹；改变这些设置需新实验。`progressive` 还固定筛选次数与晋级数量，防止已经开始正式训练后改变晋级名单；增加其 `--trials` 会产生新自动目录。`standard` / `prune` 可以增加 trials 续搜。
- test 失败不需要重训。原命令可恢复；也可用 `python tune.py --test-only --output 实验根目录` 补测所有已完成的正式候选，跳过 `PRUNED`。补测不修改排名，不测试 `screen/`。
- 首先建议 `--dry-run` 核对预算。本文预算与比例是起点，小尺寸最优不保证大尺寸最优，晋级多个配置与基线就是为了验证这一点。

## 原有 standard 模式：直接运行

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

默认 `--test-policy every-trial`：每个候选训练、验证完成后，测试自己的 best.pt，逐次保存四项 test 指标。显式传 `--test-policy final` 可只测试最终胜出的 n/s。选优始终依据 val mAP50-95，test 不参与调参。

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

只训练指定模型：`python tune.py --data 数据集/data.yaml --models yolo11s.pt`；两个模型用 `--models yolo11n.pt yolo11s.pt`。不指定则训练原四个模型。模型选择参与自动命名；切换模型组合时不复用旧的显式输出目录。`--trials` 仍是每个模型的试验总次数。

## 补齐旧实验的 test 指标

在原训练服务器运行，保留数据和各候选的 best.pt：

```bash
git pull --ff-only
python tune.py --test-only --output runs/animals_search_batch --device 0 --eval-batch 32 --workers 4
```

将 output 换成实际实验目录。自动读取 run.json 的数据路径，兼容旧版 models/<模型>/trial_0000 和新版 v11n/t000；旧搜索空间也可以补测。仅测试已完成训练与验证的候选，不下载、不训练、不重跑 val、不增删 Optuna trial。完整且权重校验一致的已有 test 直接复用；缺失、失败或四项指标不全则重测。补测可中断，同一命令继续。

可用 `--models yolo11n.pt yolo11s.pt` 限定模型，或加 `--dry-run` 查看待补数量。默认使用原数据配置；`--data` 可指定文件，但 YAML 与转换报告的内容指纹必须保持一致。数据内容原地变化仍无法通过配置指纹检测。不要拿另一个测试集的结果混入旧实验。

补测更新 REPORT.md、iterations.csv、summary.json、rounds 和 best/selection.json，不改选优规则或重新导出 best 权重。test_backfill.json 记录失败原因、未完成而跳过的候选、剩余待补项。缺少 best.pt 无法补测，脚本会记录后继续其他候选；未完成的训练需原命令恢复。资源故障则停止本次补测。补测报告不声明原训练预算已完成。

旧实验继续训练时仍应显式保留其原 `--test-policy`；默认策略变化会改变自动实验名。只补指标请使用 test-only，不要为此新建训练实验。

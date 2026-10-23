# 动物检测模型训练第一阶段交付报告

交付日期：2026年10月23日。实验：`animals_search_batch`。

## 1. 交付结论

**主交付选用 YOLO11s，轻量备选选用 YOLO11n。** YOLO11s 在本次四个模型对比中验证指标最高，测试集 mAP50-95 为 **62.92%**、mAP50 为 **72.20%**。YOLO11n 的测试集 mAP50-95 为 **58.61%**，权重更小，可作为后续设备适配的备选。

本阶段完成的是 **80 类通用动物检测模型**：识别图片中动物的类别和位置。使用最初的 Animals Detection 数据集，**不包含后续 LILA 合并数据**。露营夜间视频、连续跟踪和“某时某物种在周围徘徊”的事件总结，留到下一阶段验证与开发。

当前归档有 26 个已完成且测试成功的候选；另有 YOLO11n 第 7 次候选仍标记为 RUNNING，没有完整评估结果，本报告不计入。以下“最佳”均指当前已完成候选中的最佳。

## 2. 数据量与类别

数据来源：Kaggle Animals Detection Images Dataset（`antoreepjana/animals-detection-images-dataset`）。训练读取转换后的 YOLO 数据集。

| 划分 | 图片数 | 标注框数 | 用途 |
| --- | ---: | ---: | --- |
| 训练集 | 18,404 | 本次归档未提供完整统计 | 学习模型参数 |
| 验证集 | 3,245 | 4,271 | 比较候选、选择最佳模型 |
| 测试集 | 6,003 | 7,576 | 评估选中模型的表现 |
| **合计** | **27,652** | 未统计 | — |

图片数取自训练/评估日志；每类图片数不能直接相加当作总图片数，因为同一张图可能有多个类别。训练和验证缓存记录均为 0 背景图、0 损坏图。现有归档未包含原始标签，无法补出训练集逐类数量，也未在本次交付中重新核验数据泄漏。

**全部 80 个类别如下**（中文用于阅读，英文为模型原标签）：

| 类别 | 类别 | 类别 | 类别 |
| --- | --- | --- | --- |
| 熊（bear） | 长颈鹿（giraffe） | 蛾与蝴蝶（moths and butterflies） | 鲨鱼（shark） |
| 棕熊（brown bear） | 山羊（goat） | 鼠（mouse） | 绵羊（sheep） |
| 公牛（bull） | 金鱼（goldfish） | 骡（mule） | 虾（shrimp） |
| 蝴蝶（butterfly） | 鹅（goose） | 鸵鸟（ostrich） | 蜗牛（snail） |
| 骆驼（camel） | 仓鼠（hamster） | 水獭（otter） | 蛇（snake） |
| 金丝雀（canary） | 港海豹（harbor seal） | 猫头鹰（owl） | 麻雀（sparrow） |
| 毛毛虫（caterpillar） | 刺猬（hedgehog） | 大熊猫（panda） | 蜘蛛（spider） |
| 牛（cattle） | 河马（hippopotamus） | 鹦鹉（parrot） | 鱿鱼（squid） |
| 蜈蚣（centipede） | 马（horse） | 企鹅（penguin） | 松鼠（squirrel） |
| 猎豹（cheetah） | 美洲豹（jaguar） | 猪（pig） | 海星（starfish） |
| 鸡（chicken） | 水母（jellyfish） | 北极熊（polar bear） | 天鹅（swan） |
| 螃蟹（crab） | 袋鼠（kangaroo） | 兔（rabbit） | 蜱虫（tick） |
| 鳄鱼（crocodile） | 考拉（koala） | 浣熊（raccoon） | 老虎（tiger） |
| 鹿（deer） | 瓢虫（ladybug） | 渡鸦（raven） | 陆龟（tortoise） |
| 鸭（duck） | 豹（leopard） | 小熊猫（red panda） | 火鸡（turkey） |
| 鹰（eagle） | 狮子（lion） | 犀牛（rhinoceros） | 龟类（turtle） |
| 大象（elephant） | 蜥蜴（lizard） | 蝎子（scorpion） | 鲸（whale） |
| 鱼（fish） | 猞猁（lynx） | 海狮（sea lion） | 啄木鸟（woodpecker） |
| 狐狸（fox） | 喜鹊（magpie） | 海龟（sea turtle） | 蠕虫（worm） |
| 青蛙（frog） | 猴（monkey） | 海马（seahorse） | 斑马（zebra） |

保留了原数据集的类别定义：例如 bear 与 brown bear/polar bear、fish 与 goldfish、turtle 与 sea turtle/tortoise 同时存在。本阶段没有改成后续露营类别体系，**没有 wolf 和 wild_boar 标签**；pig 不能直接等同于野猪。

## 3. 训练与选优方式

- 对比 YOLOv8n、YOLOv8s、YOLO11n、YOLO11s；各模型从预训练权重开始，独立进行 Optuna 搜索。
- 完成次数：YOLOv8n 7 次、YOLOv8s 7 次、YOLO11n 6 次、YOLO11s 6 次。
- 输入尺寸 640；每次最多 150 epoch，连续 20 epoch 无改善早停；seed=42，开启 AMP。
- 搜索 batch=64/128/256、AdamW/SGD、学习率、权重衰减和增强参数。本实验使用旧版搜索空间 v2，首轮是 AdamW 基线，不是后来增加的 auto 基线。
- 按 **验证集 mAP50-95** 选优，再报告该候选的测试结果；没有把不同候选的最高单项指标拼在一起。每次候选均记录过测试结果，后续不应根据这些测试分数反复调参。
- 训练环境：RTX PRO 6000 Blackwell Server Edition；Python 3.11.14、PyTorch 2.10.0+cu128、Ultralytics 8.4.182。

## 4. 四个模型的最佳结果

指标越高越好。Precision 表示检出的目标有多少是对的，Recall 表示真实目标有多少被检出；mAP50-95 综合多个框重叠要求，是本次主要比较指标，不能理解成“识别正确率”。

### 验证集结果 用于选优

| 模型 | 最佳候选 | Precision | Recall | mAP50 | mAP50-95 | 较同模型首轮提升 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | trial_0005 | 76.07% | 74.20% | 80.88% | 68.13% | +2.10 个百分点 |
| yolov8s | trial_0006 | 81.75% | 74.41% | 82.24% | 70.69% | +1.54 个百分点 |
| yolo11n | trial_0004 | 76.00% | 74.37% | 80.72% | 68.73% | +0.85 个百分点 |
| yolo11s | trial_0002 | 80.48% | 78.91% | 83.89% | 72.42% | +1.40 个百分点 |

### 同一候选的测试集结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 | 定位 |
| --- | ---: | ---: | ---: | ---: | --- |
| yolov8n | 61.43% | 64.90% | 67.67% | 57.07% | 对照模型 |
| yolov8s | 71.63% | 65.28% | 70.12% | 60.44% | 对照模型 |
| yolo11n | 67.65% | 63.05% | 68.44% | 58.61% | **轻量备选** |
| yolo11s | 69.42% | 68.68% | 72.20% | 62.92% | **主交付** |

按尺寸比较，n 档选择 YOLO11n，s 档选择 YOLO11s。主交付 YOLO11s 比轻量备选的测试 mAP50-95 高 **4.31 个百分点**；但其验证指标为 72.42%、测试指标为 62.92%，相差 **9.50 个百分点**，后续应重点检查泛化表现。

## 5. 最终权重与配置

| 项目 | 主交付 YOLO11s | 轻量备选 YOLO11n |
| --- | --- | --- |
| 权重 | [best_s.pt](../../runs/animals_search_batch/best/best_s.pt) | [best_n.pt](../../runs/animals_search_batch/best/best_n.pt) |
| 获胜候选 | trial_0002 | trial_0004 |
| 文件大小 | 18.34 MiB | 5.27 MiB |
| 优化器 | SGD | AdamW |
| batch | 128 | 64 |
| 初始学习率 | 0.00439397 | 0.000170261 |
| weight_decay | 0.000212943 | 0.000482007 |
| 训练记录轮数 | 92 | 61 |

权重均已校验 SHA-256，与获胜候选记录一致。训练记录轮数不是 best.pt 所在轮数；例如 YOLO11s 在第 92 轮早停，日志记录最佳轮次为第 72 轮。完整参数见对应候选的 config.json 和 training/args.yaml。

本次测试日志中的纯推理耗时为 YOLO11s 约 0.70 ms/图、YOLO11n 约 0.38 ms/图（该 GPU、评估 batch=64）。这不是摄像头端到端延迟，也不能据此承诺实际部署帧率。

## 6. 露营相关类别的实际表现

以下为主交付 YOLO11s 的测试集结果：

| 类别 | 测试图片数 | mAP50-95 | Recall |
| --- | ---: | ---: | ---: |
| 蛇（snake） | 213 | 80.03% | 92.69% |
| 狐狸（fox） | 69 | 86.52% | 86.11% |
| 鹿（deer） | 177 | 74.19% | 84.25% |
| 浣熊（raccoon） | 51 | 73.17% | 74.44% |
| 松鼠（squirrel） | 68 | 77.15% | 88.57% |
| 熊（bear） | 39 | 42.98% | 45.24% |
| 棕熊（brown bear） | 39 | 80.56% | 79.82% |
| 北极熊（polar bear） | 55 | 90.90% | 91.23% |
| 蝎子（scorpion） | 44 | 86.34% | 90.91% |
| 蜘蛛（spider） | 207 | 89.38% | 97.18% |
| 蜱虫（tick） | 1 | 17.91% | 0.00% |

蛇、狐狸、蜘蛛等类别已有较好的测试表现；泛类 bear 的召回率只有 45.24%，不能认为所有熊类都已经识别稳定。蜱虫只有 1 张测试图且本次报告召回率为 0，证据不足。金鱼、鱿鱼和龟类的测试 mAP50-95 分别仅为 7.32%、6.10% 和 0.25%，整体均值掩盖了这些弱项。

## 7. 下一阶段重点

1. 引入已整理的 LILA 露营数据，统一泛类与细类标签，补齐狼、野猪等本阶段未覆盖的类别。
2. 在真实露营视频中检查夜间、远距离、小目标、遮挡以及空场景误报，单独统计重点物种的漏检。
3. 在目标设备上比较 YOLO11s 与 YOLO11n 的速度和内存；再加入跟踪与时间事件总结。

**阶段结论：交付 YOLO11s 作为后续联调主模型，YOLO11n 作为轻量备选；当前完成通用动物检测基线，尚未完成露营实景验收。**

## 交付明细与依据

- [80 类中英文名称与逐类测试指标](phase1_class_metrics.csv)：包含验证/测试图片数、框数及四个最佳模型的逐类测试 mAP50-95。
- [关键结果与权重校验记录](phase1_evidence.json)：保留选优结果、参数和哈希，便于核对。
- 原始依据：[实验配置](../../runs/animals_search_batch/run.json)、[汇总结果](../../runs/animals_search_batch/summary.json)、[候选明细](../../runs/animals_search_batch/iterations.csv)、[权重选择记录](../../runs/animals_search_batch/best/selection.json)。
- 数据量依据：`models/yolo11s/trial_0002/train.log`、`val_result.json`、`test.log`、`test_result.json`。原始实验与权重保存在本地 runs 目录，不随报告提交到 GitHub。

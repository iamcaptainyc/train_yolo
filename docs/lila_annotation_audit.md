# 五个 LILA 数据集检测框核查

核查日期：2026-10-20。以下计数为初次审计快照；后续版本补充排除 Caltech 的 `car` 等非动物类别，最新数量以重新运行生成的 selection_report.json 为准。逐条扫描当时页面链接对应的实际 COCO Camera Traps 标注，未全量下载图片。以下数字是元数据记录数，不代表逐张核实远端图片存在或标注完整。

## 是否所有图片都有框

- [SWG Camera Traps](https://lila.science/datasets/swg-camera-traps)：不是。页面明确框仅覆盖部分图片，完整 metadata 包含大量只有图片/序列分类的记录；使用单独的 `with_species` 框标注。
- [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)：不是。分类元数据覆盖全体图片，检测框在另外的 JSON 中，且含大量空图。
- [ENA24](https://lila.science/datasets/ena24detection)：公开动物检测数据有框，框嵌在 metadata；本次 metadata 的全部图片 ID 都关联到 bbox 字段，但仍要排除人类、vehicle 等不适用记录。没有明确 empty 类，类别 0 是 Bird，不能作为负样本。
- [Oregon Critters](https://lila.science/datasets/oregon-critters/)：不是所有图片都有框。官方区分 91,045 张有框图片和 8,864 张空图；框嵌在 metadata。本次还排除了无效动物框和不可用记录。
- [Nkhotakota Camera Traps](https://lila.science/datasets/nkhotakota-camera-traps/)：不是。只有子集有框，另有明确空图和仅有分类的动物图片；框嵌在 metadata。当前文件记录数与页面历史汇总不同，以实际文件筛选。

## 实际标注筛选结果

`bbox 记录图片` 表示出现物体 bbox 字段且能关联到 metadata 的图片数，包含待排除或无效框；不等于最终正样本。`明确空图池` 排除了矛盾标注和不可用记录。最终正样本要求有效动物框，任一无效动物框会导致该图片被排除。

| 数据集 | metadata 图片 | bbox 记录图片 | 有效正样本 | 明确空图池 | 入选负样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SWG Camera Traps | 2,039,657 | 88,135 | 87,038 | 264,610 | 4,695 |
| Caltech Camera Traps | 243,100 | 61,945 | 61,762 | 125,562 | 3,365 |
| ENA24 | 9,676 | 9,676 | 8,652 | 0 | 0 |
| Oregon Critters | 99,909 | 91,045 | 87,368 | 8,864 | 4,712 |
| Nkhotakota Camera Traps | 324,635 | 40,743 | 37,772 | 164,139 | 2,101 |

最终 **282,592 张正样本 + 14,873 张负样本 = 297,465 张**，负样本比例约 **5.000%**。默认 seed=42；将不足配额从其他数据集的明确空图池补齐，因此 ENA24 没有负样本，其他数据集比例不必各自恰好 5%。单独下载 ENA24 时会如实报告负样本不足。

负样本必须有明确空图标签；仅缺 bbox、缺标注、有动物但无框都不会入选。元数据的空图标签仍可能存在人工标注错误，不能视为逐张视觉复核的结果。

## 本次使用的标注文件

文件名与 SHA-256 如下，供比较后续版本。目录由 [LILA 官方目录](https://lila.science/wp-content/uploads/2023/06/lila_camera_trap_datasets.csv) 和上述页面解析，图片下载仍保留原始相对路径。

| 数据集 | 文件 | SHA-256 |
| --- | --- | --- |
| SWG Camera Traps | `swg_camera_traps.zip` | `b3e3daed77dd8c4ab971477cde2044a97ed74380180ca020f73ad4049914e34f` |
| SWG Camera Traps | `swg_camera_traps.bounding_boxes.with_species.zip` | `2aa241381f6949e39e9a86108ad1772555a69402c684ceb703ec659b259db1f4` |
| Caltech Camera Traps | `caltech_camera_traps.json.zip` | `976bd9bc9380df7a03018ac70b5340bd2ff81d79a99271058f026a539450d1d6` |
| Caltech Camera Traps | `caltech_bboxes_20200316.json` | `9cc208db67b7dcf3fa403920c203a416527628f76a6b838371ab654478bf2ea3` |
| ENA24 | `ena24.json` | `8b426c87c948ab91dda10e5b210fc469db8b0c095a15cc7c03bb4ddceffa3160` |
| Oregon Critters | `oregon_critters.zip` | `0b649fc7ffc90df9c7dc9a8a79dadd3a387e9b7dae9be9ad189beb3ec5dcc5ac` |
| Nkhotakota Camera Traps | `nkhotakota_camera_traps.json.zip` | `5915d290527c4c565390b9c1d850a209371d8b152064aa94007bec6d2bb55671` |

## 运行与验证

```bash
python -m pip install -r requirements-lila.txt awscli
python download_lila.py --urls-file examples/lila_urls.txt --audit-only
python download_lila.py --urls-file examples/lila_urls.txt --mode parallel --dataset-workers 2 --workers 32
```

筛选报告会在每次运行时重新生成，版本变化时数量可能不同。正式下载时返回不存在/损坏的图片会记入 `selection/failed_images.json`，返回失败状态，重跑补下载。只有实际下载全部成功，才能认为筛选清单已完整落盘。

本次验证：全量扫描五个数据集的标注并生成最终清单；AWS 实际下载 Caltech 一张正样本和一张空图并通过图片格式校验；66 项单元测试通过，涵盖无框动物排除、类别 0 不误判、批次配额补齐、COCO 图片引用、部分图片续传和失败记录。没有执行全量图片下载或 YOLO 训练。

COCO 子集仍保留各数据集原始类别，可使用 [prepare_lila.py](lila_yolo.md) 统一类别、转换 YOLO 并采用相机/序列分组划分，避免相邻帧跨训练和验证集。现有 Animals Detection 转换器不能直接处理这些 COCO 标注。已入选图片有框并不保证其中全部动物都已被完整标注。

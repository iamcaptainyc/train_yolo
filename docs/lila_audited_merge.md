# 五个 LILA 数据集：合并、类别层级处理和框图审计

正式入口是 `prepare_lila_audited.py`。它复用原有转换器的相机/序列分组，新增原始标注全量统计、官方分类树合并、逐图逐框独立审计和 Markdown 框图报告。原来的 `prepare_lila.py` 仍保留旧的名称合并行为；需要解决层级混淆时请用新入口。

## 直接运行

在服务器仓库目录执行：

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt PyYAML

python prepare_lila_audited.py \
  --source datasets/lila \
  --output datasets/lila_yolo_audited \
  --val-fraction 0.2 --seed 42 \
  --copy-mode auto --previews-per-class 1
```

默认就是下面五个数据集，无需查名字或再传网址：

| 页面 | 已下载目录 |
| --- | --- |
| [SWG](https://lila.science/datasets/swg-camera-traps) | `datasets/lila/swg-camera-traps` |
| [Caltech](https://lila.science/datasets/caltech-camera-traps) | `datasets/lila/caltech-camera-traps` |
| [ENA24](https://lila.science/datasets/ena24detection) | `datasets/lila/ena24` |
| [Oregon](https://lila.science/datasets/oregon-critters/) | `datasets/lila/oregon-critters` |
| [Nkhotakota](https://lila.science/datasets/nkhotakota-camera-traps/) | `datasets/lila/nkhotakota-camera-traps` |

每个目录需要保留下载脚本产生的 `images/`、`annotations/`、`selection/selected_annotations.json` 和 `selection/selection_report.json`。无需重新下载图片。程序会读取原始标注来核对下载清单，不能仅用筛选后的标注声称统计了“原始全集”。缺少这些文件时会停止。

默认优先硬链接，跨文件系统则复制；同盘硬链接不再占用一份图片数据空间，但输入和输出共享内容，运行后勿修改原图。`--copy-mode copy` 需要完整图片副本空间。统计去重使用临时 SQLite；图片校验和输出哈希会全量读取图片，耗时与真实 I/O 和解码速度有关，各阶段显示进度。

输出目录必须为空。中断后源数据不变，但部分输出不能用于训练，修复问题后换一个新的 `--output`。只有全部审计通过，才写入最终 `data.yaml`；不是可断点恢复的转换任务。

## 类别层级怎么处理

固定使用仓库内 `configs/lila_taxonomy.csv`，来自 [LILA 官方分类映射](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/) 所链接的 CSV。快照提取五个数据集的原始映射行，下载日期、完整 CSV 指纹和子集指纹记录在 `configs/lila_taxonomy.source.json`。不在运行时自动更新，避免不同运行悄悄换分类标准。

1. 类别 ID 始终属于数据集和它自己的 bbox 文件；metadata 文件的同一数字 ID 不参与错误匹配。
2. 相同的科学分类 ID 合并，别名不依赖字符串相似度推断。
3. 若存在父类框，则该父类下面的已标注物种一起上卷。例如 `bird` 与 `american crow` 最终都归入 `class:aves`；有 `rodent` 时，松鼠、鼠等也会被统一到 `order:rodentia`。这是明确舍弃细分类，以换取互斥且一致的监督。
4. 最终科学分类节点不会同时存在祖先和后代。只使用实际入选的有框类别决定层级，零样本类别不会把整个树拉粗。
5. `animalia`、`mammalia`、`superorder` 等过宽标签默认排除，避免少数 `other animal` / `small mammal` 把所有动物收成一类。**含被排除类的整张图片都丢弃**，同图其他类别也会损失样本，不会把这类物体隐去后当背景。
6. 默认负样本沿用下载清单，不重新抽样。排除整图与分组后总负样本比例不保证仍恰为 5%，报告会列实际数量。

这不是绝对正确的生物学分类证明。官方映射也明确说明复合类会丢失信息，例如 bird of prey 被近似映射为 birds；官方表还可能存在映射错误。输出采用 `rank:scientific_name`，不把科/目标签伪装成具体物种。每个原类、官方对应、最终父类、合并/排除原因都会写入 `taxonomy_decisions.json` 和报告。

如需排除某个有疑问的来源类，先从报告读取真实 ID，再用（示例中的 ID 需替换）：

```bash
python prepare_lila_audited.py --source datasets/lila --output datasets/lila_yolo_reviewed \
  --exclude-category "swg-camera-traps:实际ID"
```

可重复传入该参数。新版本原始类别若在快照中找不到，会明确报错，不猜物种；需要人工核对后用 `--taxonomy-file` 指向同格式的已审阅 CSV。

## 框是否真的对应：检查链

**A. 原始文件 → 下载清单（全量）**

- 从 selection_report 定位原始 metadata 与 bbox 文件，ZIP 使用已有安全 JSON 缓存。
- 每张入选图核对原始 image_id 和原始相对路径，保留 source_image_id。拒绝重复 ID/文件名和悬空 annotation。
- 每个正样本的完整 category_id + COCO xywh 多重集合必须与原 bbox 文件精确相同。换图、换 ID、改框、漏框或额外框均拒绝。
- 负样本必须从原始 metadata 找到明确 empty 标签，不能有非 empty 标注，也不能关联原始物体框；不能用“没找到框”代替空图证据。

**B. 图片与坐标 → 实际 YOLO 文件（全量）**

- 解码每张原图、检查 EXIF 方向和实际尺寸与标注尺寸。读取 EXIF 后重新打开 PNG 做 verify，避免流被消耗导致误报。
- 输出原图不缩放、不转码；独立计算每个输出文件 SHA-256，与输入图指纹逐一比较。
- 重新读取真正写到磁盘的 `.txt`，检查 class ID、数值范围、标签数，再反算像素框，与正确类别的原始框逐一配对。
- 8 位小数产生的允许像素误差仅为 `max(W,H)*1e-8 + 1e-7`。报告记录实际最大误差，每框记录自己的误差。
- 原始相同类别、完全相同位置的重复框去重，单独统计。若多个原类上卷后变成同类同框，也同样记录去重。
- 默认拒绝需要边缘裁剪的标注；只有显式 `--allow-one-pixel-clipping` 才允许原转换器规定的至多 1 像素越界裁剪，并报告原始裁剪量及裁剪后反算误差。不能把裁剪称为“完全没改原框”。
- 同时核验入选图片总数 = 保留数 + 整图排除数；保留图片中，正负角色必须与标签是否为空一致。

**C. 框图与人工复核（分层样本）**

报告包含并排图：左边红框来自已经核对过原文件的 COCO 框，右边绿框来自实际 YOLO txt 的反算结果。覆盖每个保留的原始类别、每个可用的数据集/划分/正负角色，并额外选择最多框和最小框图片；允许裁剪时还补充裁剪样本。`--previews-per-class 3` 可以增加覆盖。

预览只用于抽样观看；数值与文件对应检查覆盖所有保留图片和框。每个框可追溯到原始 annotation ID、筛选 annotation ID、原类别、原 xywh、反算 xyxy、标签文件行号和像素误差。

任何程序都不能保证原作者把动物位置画对、物种认对、所有动物都标全，或人工标记的 empty 图绝对没有动物。审计证明的是输入→输出转换关系；这些原始标注质量问题仍需看框图或人工复核。

## Markdown 报告内容

运行成功后打开 `datasets/lila_yolo_audited/AUDIT_REPORT.md`：

- 五数据集原始图片/标注/框总数、筛选正负样本数、整图排除数、最终图片和框数。
- 每个原始 metadata 类别的原 ID、名称、标注记录数、去重图片数、框记录数。
- 每个原始 bbox 类别的框数与图片数、筛选框数、保留原框数、对应最终类及原因。
- 合并后连续 YOLO ID、科学分类、总/train/val 框数与图片数、全部原始来源类别。
- 相机/序列/连通分量/图像 SHA-256 的跨集交集检查、缺席验证类别、保守留训练的数据集。
- 全量校验数量、最大像素误差、裁剪量、重复框、重复图片记录数、唯一图片 SHA-256 数与证据框图。

所有计数来自**本次服务器实际文件**，不是先填一个历史数量。图片多标签时各类图片数相加可能大于总图片数。字节相同且标注一致的图片保留为独立记录，但连接到同一划分，避免跨集精确重复；不会检测不同编码/相似帧的视觉重复。

## 输出文件

```text
datasets/lila_yolo_audited/
  data.yaml                       # 所有审计成功后才生成
  images/{train,val}/<dataset>/...
  labels/{train,val}/<dataset>/...
  AUDIT_REPORT.md                  # 详细类别统计、合并规则、审计数值与框图
  previews/<dataset>/*.jpg         # 原框/实际 YOLO 框并排图
  preview_index.json
  box_audit.jsonl                  # 全量逐图、逐框来源、坐标、哈希与误差
  audit_sources.json               # 原始全集统计、文件路径与指纹、代码指纹
  taxonomy_decisions.json          # 原类→官方科学分类→最终类别与理由
  excluded_images.jsonl            # 整图排除清单及触发类别
  class_mapping.json
  split_manifest.jsonl
  report.json
```

程序只新建 train/val，暂不生成独立 test。因此评估时用 `--split val`；需要独立测试集的四模型搜索脚本不可直接使用此 YAML。

```bash
python train.py --data datasets/lila_yolo_audited/data.yaml \
  --model weights/yolo11s.pt --epochs 100 --batch 64 --workers 4 --device 0
```

ENA24 无可靠分组元数据时整体留在 train；其他数据若也有缺失同样处理，不能凭文件名臆测相机。有补充相机信息可沿用 `--group-map cameras.json`，格式见 [原分组说明](lila_yolo.md)。

原始标注移动到其他机器后，程序尝试在各数据集 `annotations/` 中按原 basename 唯一定位；若仍找不到或文件重名，可传：

```json
{
  "caltech-camera-traps": {
    "metadata": "/你的路径/caltech_camera_traps.json.zip",
    "bbox": "/你的路径/caltech_bboxes_20200316.json"
  }
}
```

保存为 `original_files.json` 后添加 `--original-files original_files.json`，其他未覆盖数据集仍使用下载报告中的路径。

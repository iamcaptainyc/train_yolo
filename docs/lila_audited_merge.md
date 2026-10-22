# 五个 LILA 数据集：合并、类别层级处理和框图审计

正式入口是 `prepare_lila_audited.py`。它复用原有转换器的相机/序列分组，新增原始标注全量统计、官方分类树合并、逐图逐框独立审计和 Markdown 框图报告。原来的 `prepare_lila.py` 仍保留旧的名称合并行为；需要解决层级混淆时请用新入口。

## 直接运行

在服务器仓库目录执行：

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt PyYAML

python prepare_lila_audited.py \
  --source datasets/lila \
  --output datasets/lila_yolo_90505 \
  --val-fraction 0.05 --test-fraction 0.05 --seed 42 \
  --copy-mode hardlink --workers 16 --previews-per-class 1
```

审计入口默认按 **train/val/test = 90%/5%/5%** 的目标比例联合分配完整连通组。`--val-fraction`、`--test-fraction` 分别指定验证和测试比例，训练比例为余数；两者之和必须小于 1。

默认就是下面五个数据集，无需查名字或再传网址：

| 页面 | 已下载目录 |
| --- | --- |
| [SWG](https://lila.science/datasets/swg-camera-traps) | `datasets/lila/swg-camera-traps` |
| [Caltech](https://lila.science/datasets/caltech-camera-traps) | `datasets/lila/caltech-camera-traps` |
| [ENA24](https://lila.science/datasets/ena24detection) | `datasets/lila/ena24` |
| [Oregon](https://lila.science/datasets/oregon-critters/) | `datasets/lila/oregon-critters` |
| [Nkhotakota](https://lila.science/datasets/nkhotakota-camera-traps/) | `datasets/lila/nkhotakota-camera-traps` |

每个目录需要保留下载脚本产生的 `images/`、`annotations/`、`selection/selected_annotations.json` 和 `selection/selection_report.json`。无需重新下载图片。程序会读取原始标注来核对下载清单，不能仅用筛选后的标注声称统计了“原始全集”。缺少这些文件时会停止。

脚本默认 `auto` 优先硬链接，跨文件系统则复制；上面的命令显式使用 `hardlink`，不支持时直接报错，确保不会复制图片。硬链接不再占用一份图片数据空间，但输入和输出共享内容，运行后勿修改原图。`--copy-mode copy` 需要完整图片副本空间。统计去重使用临时 SQLite；图片校验和输出哈希会全量读取图片，耗时与真实 I/O 和解码速度有关，各阶段显示进度。

`Validate and hash`、`Write YOLO`、`Independent output audit` 现在使用有界线程池，默认 `--workers 8`，上例使用 16。最多保留约 `2 × workers` 个已提交/缓冲任务，不会一次排队几十万张；解码图片同时驻留的数量受线程数限制。`--workers 1` 恢复串行，线程数不改变标签、划分或逐框审计内容。进度按完成任务更新，结果按输入顺序汇总。

提速依赖磁盘吞吐、图片编码与 CPU；110 核不意味着应开 110 个线程。可先用 16，磁盘空闲且内存充足再试 32，内存或 I/O 压力大则降为 8。此处 workers 属于转换脚本，与训练 DataLoader 的 workers 是独立设置。首次完整解码、哈希和核验；之后按批次断点跳过未变化且已完成批次的解码检查，不保存逐图正常结果。预览框图与报告汇总按序执行，保证输出顺序可复现。

新版默认自动恢复：中断后再次执行原命令，保持相同 `--output` 即可。Ctrl+C 会取消排队任务并等待正在执行的任务退出；每 128 张组成一个批次，完整批次的进度已经落盘；恢复时不再解码未变化的已完成批次，未完成批次最多重检 128 张。只有所有保留图片通过审计才生成 `data.yaml`；开始恢复时会撤下上次的训练 YAML 与正式报告，避免误用旧证书。

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
- 8 位小数产生的允许像素误差仅为 `max(W,H)*1e-8 + 1e-7`。报告记录实际最大误差；每框误差在内存中核验，默认不保存正常框的逐条明细。
- 原始相同类别、完全相同位置的重复框去重，单独统计。若多个原类上卷后变成同类同框，也同样记录去重。
- 默认自动将各边至多 1 像素的越界裁回图内，报告裁剪量及裁剪后反算误差；更大的越界、退化框或非法坐标会排除整张图片。`--strict` 恢复遇错停止并拒绝裁剪的行为；旧参数 `--allow-one-pixel-clipping` 仍兼容。不能把裁剪称为“完全没改原框”。
- 同时核验入选图片总数 = 保留数 + 整图排除数；保留图片中，正负角色必须与标签是否为空一致。

**C. 框图与人工复核（分层样本）**

报告包含并排图：左边红框来自已经核对过原文件的 COCO 框，右边绿框来自实际 YOLO txt 的反算结果。覆盖每个保留的原始类别、每个可用的数据集/划分/正负角色，并额外选择最多框和最小框图片；允许裁剪时还补充裁剪样本。`--previews-per-class 3` 可以增加覆盖。

预览只用于抽样观看；数值与文件对应检查覆盖所有保留图片和框。每个框可追溯到原始 annotation ID、筛选 annotation ID、原类别、原 xywh、反算 xyxy、标签文件行号和像素误差。

任何程序都不能保证原作者把动物位置画对、物种认对、所有动物都标全，或人工标记的 empty 图绝对没有动物。审计证明的是输入→输出转换关系；这些原始标注质量问题仍需看框图或人工复核。

## Markdown 报告内容

运行成功后打开 `datasets/lila_yolo_90505/AUDIT_REPORT.md`：

- 五数据集原始图片/标注/框总数、筛选正负样本数、整图排除数、最终图片和框数。
- 每个原始 metadata 类别的原 ID、名称、标注记录数、去重图片数、框记录数。
- 每个原始 bbox 类别的框数与图片数、筛选框数、保留原框数、对应最终类及原因。
- 合并后连续 YOLO ID、科学分类、总/train/val/test 框数与图片数、全部原始来源类别。
- 三组的目标与实际比例、各数据集在三组的图片/负样本/框数，以及 train↔val、train↔test、val↔test 的相机/序列/连通分量/图像 SHA-256 交集检查、缺席类别、保守留训练的数据集。
- 全量校验数量、最大像素误差、裁剪量、重复框、重复图片记录数、唯一图片 SHA-256 数与证据框图。

所有计数来自**本次服务器实际文件**，不是先填一个历史数量。图片多标签时各类图片数相加可能大于总图片数。字节相同且标注一致的图片保留为独立记录，但连接到同一划分，避免跨集精确重复；不会检测不同编码/相似帧的视觉重复。

## 输出文件

```text
datasets/lila_yolo_90505/
  data.yaml                       # 所有审计成功后才生成
  images/{train,val,test}/<dataset>/...
  labels/{train,val,test}/<dataset>/...
  AUDIT_REPORT.md                  # 详细类别统计、合并规则、审计数值与框图
  previews/<dataset>/*.jpg         # 原框/实际 YOLO 框并排图
  preview_index.json
  repairs.jsonl                    # 仅记录自动修正的图片、框数与最大裁剪量
  validation_errors.jsonl          # 仅异常图片/标注，完整列表
  .lila-state/progress.json        # 每批完成标记与批次摘要，无逐图记录
  .lila-state/run.json             # 配置与运行状态
  audit_sources.json               # 原始全集统计、文件路径与指纹、代码指纹
  taxonomy_decisions.json          # 原类→官方科学分类→最终类别与理由
  excluded_images.jsonl            # 整图排除清单及触发类别
  class_mapping.json
  report.json
```

程序现在生成 train、val、test 和包含三个路径的 `data.yaml`，可直接用于四模型搜索。建议 `--test-policy final`，仅在模型和参数确定后评估测试集，选优仍使用验证集 mAP50-95。

90%/5%/5% 是图片数量的目标，不能保证精确命中：相机、序列和重复图片的连通组永不拆开，缺可靠分组的数据集整体留训练，所有类别都至少保留训练样本。先为 val/test 联合预留独立正样本组，再按整体与各数据集目标分配剩余完整组；可用组不足或无法同时保留训练类别时会失败，不退回随机拆帧。报告列出实际比例，罕见类不保证同时出现在 val 和 test。

划分发生变化，需要选择新的输出目录；不要混用旧版两组的 YAML、标签或分组清单。

```bash
python train.py --data datasets/lila_yolo_90505/data.yaml \
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

最终测试命令（权重路径替换为你的最佳模型）：

```bash
python train.py --mode val --split test --data datasets/lila_yolo_90505/data.yaml \
  --model /你的模型/best.pt --device 0
```

正在运行的旧进程不会自动加载新代码。先 Ctrl+C，再更新代码并重跑。旧版本没有保存校验成功记录，无法从终端显示的前 20 条错误恢复此前所有校验结果：原始旧版本未保存过结果时，需要首次建立批次进度。若已有上一版逐图 SQLite 缓存，会将仍有效的记录归并成批次进度，再删除旧大文件。旧输出若为空可以继续使用；旧版已有部分输出且没有 `.lila-state/run.json` 时，换一个新的输出目录；不要删除原始下载目录。

## 默认自动处理与恢复

默认无需交互确认：

- 损坏/截断、无法解码、缺失的图片自动排除；不尝试用宽松解码掩盖损坏。
- 实际尺寸与标注不符、EXIF 未归一化、不可安全修复的标注，整张图片排除。
- 至多 1 像素的轻微越界自动裁剪；完全同类同位置重复框自动去重。相同图片内容却有冲突标注时，相关版本全部排除，不猜哪一份正确。
- 原图不改写、不删除；“丢弃”仅表示不纳入本次 YOLO 输出，不能把有问题的正样本改为空标签负样本。
- 保留图片按相机/序列/内容哈希重新分组，以 90/5/5 为目标比例，不强行拆散组。

正式报告 `AUDIT_REPORT.md` 汇总各数据集自动排除数量、修正框数量、原始及合并类别数量和框图。所有排除的逐图原因见 `excluded_images.jsonl`，完整校验错误见 `validation_errors.jsonl`，不受终端前 20 条限制。`repairs.jsonl` 只列修正图片、修正框数与最大裁剪量；`report.json` 的 `automatic_exclusions` 保存异常明细，`cache` 仅为汇总计数。正常图片不会生成逐图校验或逐框记录。

如果所有正样本都不可用、相机组不足、原始标注来源不一致、分类表缺失或磁盘写入失败等，不能安全生成可训练数据：会输出 `PROCESSING_REPORT.md` 说明原因并返回失败，保留检查点，绝不伪造成功。`write_errors.jsonl`、`audit_errors.jsonl` 保存对应文件错误。

默认只保留 `.lila-state/progress.json` 中每批（128 张）的完成标记和一个合并摘要，以及 `run.json` 的配置/运行状态。摘要基于该批文件路径、大小、mtime_ns、设备/inode、标注与算法版本计算，不包含逐图片结果、尺寸、标签或 SHA-256 清单。28.6 万张图片的两个阶段断点通常约几百 KB，而不是数百 MB 的数据库。

**可以 Ctrl+C 中断后重跑同一命令**。校验按批次执行并发任务，整批完成后原子更新进度；恢复时先检查各批文件属性与标注摘要，未变化且已完成的批次不再做完整图片解码/verify，异常图片仍重试。若某图片变化，只使其所在批次失效。中断未完成批次最多重做 128 张。一次恢复的其他步骤仍会运行：

- 为保证按内容 SHA-256 连接重复图片、防止划分泄漏，已完成图片的哈希在内存中重新计算，需要读取图片字节，但不重复完整解码，也不把哈希保存到磁盘。
- 框坐标从原标注重新计算，报告计数重新汇总；缺少原始宽高时会读取图片头部尺寸。
- YOLO 图片/标签通过实际文件判定是否可复用；独立输出审计按批次恢复，仍反算标签，已完成且未变化的审计批次省略重复图片哈希。
- 保留类别/数量统计、错误和修正列表以及抽样框图，不生成 `box_audit.jsonl`、`split_manifest.jsonl` 或正常图片 SQLite 校验库。

批次摘要依赖文件属性能反映改动；硬链接会改变 ctime，因此摘要不使用 ctime。若外部操作同时保留大小、mtime 和文件身份，不能仅凭断点发现篡改。需要强制完整重新核验时加 `--recheck-all`。不要手工修改进度文件。

从上一版详细模式升级时，脚本先读取旧校验库，把仍有效的成功记录归并成批次断点，然后删除本程序生成的 `cache.sqlite3`、WAL/SHM、`box_audit.jsonl` 和 `split_manifest.jsonl`，不会删原图、YOLO 标签或异常清单。最初没有任何缓存的旧进程无法恢复未记录的历史进度。

仅在明确需要完整逐图片证据时才使用 `--detailed-audit`，它恢复上一版占用较大的逐图缓存和明细输出。普通命令默认不启用。由轻量模式改成详细模式需选择新输出目录。

图片与标签用临时文件原子替换，恢复时根据现有图片和标签复用输出；若修复输入导致划分变化，只清理专用输出目录中符合本程序命名规则的旧图片和标签，再重新写入，避免旧样本残留到其他划分。类映射、源目录、数据集列表、分组映射、seed、划分比例或复制模式在已经写入输出后改变时，需用新输出目录，避免混合数据版本。线程数可自由调整。同一输出目录有文件锁保护，不支持同时启动两个转换进程。

```bash
# 首次运行 / 中断后恢复，使用相同命令与目录
python prepare_lila_audited.py --source datasets/lila \
  --output datasets/lila_yolo_auto --copy-mode hardlink --workers 16

# 可选：强制重新检查所有输入和输出
python prepare_lila_audited.py --source datasets/lila \
  --output datasets/lila_yolo_auto --copy-mode hardlink --workers 16 --recheck-all
```

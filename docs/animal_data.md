# 通用动物索引与逐层均衡抽样

新入口 `animal_data.py` 独立于旧版固定56类的 `extract_camping.py`。索引保存原始类别和框，`animal_catalog.json` 保存通用层级与体型标签，`camping19.yaml` 只是可替换的输出方案。修改分组/标签不重建原始索引，不强制19类或30K上限。

## 1. 建立索引

报告正文 `REPORT.md` 仅展示实际类别数、图片总数、train/val/test 图片数和逐类分集/合计统计（按图片计数，多类别同图可重复计入各类）。原始类别、层级采样、来源及校验细节单独保存为 `REPORT_APPENDIX.md`，机器明细保留在 `report.json`。预览尚未划分集合，只显示抽样类别和数量。

已经生成的数据集可直接刷新报告，无需重建索引、重新抽样或扫描图片：

```bash
python animal_data.py report --dataset yolo_datasets/animals_30k_19
```

旧报告缺少全局分集计数时，从同目录 `selection.jsonl` 汇总；此命令仅更新报告文件，不改变图片、标签或划分。

在服务器仓库目录运行，AP-10K 路径按实际下载位置替换：

```bash
python animal_data.py index \
  --index datasets/animal_index.sqlite \
  --ap10k datasets/imllti/ap-10k \
  --animals yolo_datasets/animals-detection-images-dataset \
  --lila-source datasets/lila
```

AP-10K 只读一套 split 方案（默认1），汇总该方案的 train/val/test；直接保留所有原类别，不套用旧 AP_NAMES 的排除策略。Animals 读取原80类 YOLO，不使用已经缩成56类的导出目录。LILA 读取每个子数据集的 `selection/selected_annotations.json` 与 `images/`，因此保留原始细类，不从合并后的框猜物种。

同名来源注释未变则跳过；新增来源会追加，注释变化按来源原子替换。中断只回滚正在建立的来源，已完成来源保留，重跑即可；不是逐图片断点续跑。图片发生变化时加 `--refresh`。指纹使用原 JSON/YAML 以及 YOLO 标签的路径/大小/mtime，不读取图片内容做哈希；手动伪造相同元数据的修改不能检测。

索引包含路径、原类别、原图ID、归一化框与原像素框/尺寸或原YOLO行证据、来源分组、文件元数据，以及问题记录。不会复制图片或解码检查损坏JPEG。格式错误、缺失图片、框非法等记录写入 errors，不作为训练样本；未映射类别仍完整留在索引。索引本身是有意保存的可复用标注数据，不是逐图片解码校验缓存。

## 2. 先预览

### 通用来源筛选（共用一份19类配置）

索引仍可包含所有来源。`--source-config` 只控制候选池，不改变类别方案、图片面积门槛或抽样模式。内置两个来源配置：

| 来源配置 | 允许来源 |
| --- | --- |
| `configs/sources_lila.yaml` | 仅 LILA（包含已建立索引的各 LILA 子集） |
| `configs/sources_ap10k_animals.yaml` | 仅 AP-10K 和 Animals Detection |

```bash
# 只用 LILA，最多30K、起始ID27、默认平铺硬链接及逐图修复
python animal_data.py sample \
  --index datasets/animal_index.sqlite --policy configs/camping19.yaml \
  --source-config configs/sources_lila.yaml \
  --lila-ratio 1/16 --max-images 30000 --class-id-start 27 \
  --output yolo_datasets/animals_30k_lila

# 只用 AP-10K 和 Animals Detection
python animal_data.py sample \
  --index datasets/animal_index.sqlite --policy configs/camping19.yaml \
  --source-config configs/sources_ap10k_animals.yaml \
  --source-first --max-images 30000 --class-id-start 27 \
  --output yolo_datasets/animals_30k_ap_animals
```

也可不用额外文件，直接 `--only-sources lila` 或 `--only-sources ap10k animals`；支持原始来源的精确名称，例如 `--only-sources caltech-camera-traps`。命令行覆盖来源配置，来源配置覆盖主 policy 中的 `allowed_sources`；全部未设置则使用索引中的所有来源。未知名称、空来源列表会报错，避免误选。`preview` 支持同样选项。筛选后不强行凑满30K或19类，缺少的类别会在附录列出；实际有样本类别才生成ID。

主报告新增“来源和划分”表：按来源家族（lila/animals/ap10k）汇总各分集图片、负样本及框数，并列各分集合计。以实际保留图像为准，不累加多类别图片造成重复。旧输出运行 `python animal_data.py report --dataset 输出目录` 即可补表，无需重新抽样；独立 repair 删除后也同步更新。

当前 camping19 将 cougar/puma 及其下级按应用需求归入 `leopard`，不再归入 `lion`；这是业务合并，美洲狮在生物分类上不属于狮子或豹。旧输出只刷新报告不会改变标签，使用新映射需要重新抽样，原始索引可复用。

```bash
python animal_data.py preview \
  --index datasets/animal_index.sqlite \
  --policy configs/camping19.yaml \
  --max-images 30000 --output runs/preview_animals19
```

只生成 REPORT.md/report.json，不建立图片链接。报告包含有效索引中全部原类别的图片/框数量、所有采样层级可用/抽取数量、原始细类的来源、被排除标签及索引错误。未恢复或未审核的名称不会自动猜测。

## 3. 生成 animals_30k

```bash
python animal_data.py sample \
  --index datasets/animal_index.sqlite \
  --policy configs/camping19.yaml \
  --max-images 30000 --output yolo_datasets/animals_30k_19
```

输出必须为新目录。图片硬链接并平铺到 images/train、images/val、images/test；标签对应平铺，默认ID从0开始，`--class-id-start 27` 可预留0～26。负样本默认目标5%，也计入30K硬上限；缺少足够可用样本不会重复图片凑数。无实际样本的输出类不伪造训练ID，报告显示实际结果。

`sample` 默认在写出每张选中图片时立即检查并修复（`--repair-images`），不在导出后再扫描一次目录。先建立硬链接，再调用 `repair_images` 的逐图检查：正常图不改写；可恢复 JPEG 原位重新编码，其他硬链接同步修复；完全无法解码则删除本次输出链接、不生成标签，原始来源文件保留。修复后的索引文件时间/大小同步更新，不保存正常图检查记录。`--no-repair-images` 可显式关闭，`index` 和 `preview` 不解码图片。

若原位修复导致文件大小/时间与旧索引不一致，而设备和 inode 未变，默认继续在写出循环内验证：解码正常且尺寸符合已保存的 COCO 标注尺寸后，仅更新相关索引记录，不重建全库。尺寸不一致的图片从本次输出剔除，并更新数量；问题原图保留。文件被替换（设备/inode 变化）或关闭修复时仍要求刷新索引。此检查不能识别同尺寸画面被人为替换；不会据此宣称原始标注语义得到验证。

最终数量、各类/分集计数与 selection.jsonl 均按保留样本更新，问题记录在 `image_repair/issues.jsonl`，详情在附录。坏图剔除后不重复抽样补足，因此可能少于30K；某类全部剔除时保留其ID定义（0张），正文类别数只计有样本类别。可修复表示解码正常，不代表恢复原本丢失的画面。权限或磁盘错误等非坏图问题会中止并保留 INCOMPLETE 标记，不能将这种错误当成坏图删除。

抽取时默认用 8 个进程并行检查选中的图片。可以在原 `sample` 命令后加 `--repair-workers 16` 调整并发，或用 `--repair-workers 1` 串行执行。只有检测到损坏且可恢复的 JPEG 才会重新编码。

### 单独使用 repair_images.py

对已有数据集递归检查、修复，无法解码的图片和对应标签自动删除：

```bash
python repair_images.py yolo_datasets/animals_30k_19 \
  --workers 16 --output runs/repair_animals_30k_19
```

支持多个目录：`python repair_images.py 数据集A 数据集B --workers 16 --output runs/repair_multiple`。加 `--check-only` 只检查（请使用独立日志目录）。中断后重跑同一命令继续；如果图片清单改变或要重查已完成内容，加 `--restart`。独立命令的 `--output` 是日志目录，不是图片输出目录。修复写回同一 inode，所以其他硬链接同步；删除只删除被扫描路径及对应标签，不会自动寻找并删除所有硬链接别名。避免同时训练或运行其他图片写入任务。

独立修复删除图片后自动更新数据集统计，支持本脚本的精简报告和 selection.jsonl，原统计另存备份；若独立修复改动了索引所指的原图，下次抽样前用原 index 命令加 `--refresh` 更新索引。

默认 `sampling_priority: source_first`：先保持最终大类尽量均衡，每个大类内部先取 AP-10K，再取 Animals Detection，最后用 LILA 补足；在当前来源内部再按成员和细类逐层均衡。不会为了补齐某个细类，跳过仍有可用图片的优先来源。例如 AP-10K 的狐狸足够填满小型兽类配额时，不会为加入 LILA 独有的獴或鼬而选入 LILA。只有 LILA 有样本的最终大类仍可使用 LILA，以保持大类覆盖。

同一来源内，成员有更多细类不会获得更多父级配额。粗标注占独立桶；多类别同图只存一次，计数可重叠，均衡为软目标，总图片数为硬上限。负样本也按来源优先抽取。若需恢复旧的细类覆盖优先策略，可将 policy 的 `sampling_priority` 改为 `hierarchy_first`。

`preview` 和 `sample` 均可直接开关模式：`--source-first` 开启来源优先；`--no-source-first` 恢复细类均衡优先（同一细类内仍优先 AP-10K/Animals）。命令行开关覆盖 policy；不传时使用 policy，当前 camping19 默认开启来源优先。两种模式都保持最终大类尽量均衡，均不改变 LILA 面积阈值。预览与正式抽样使用相同开关，实际模式记录在 report.json 和报告附录中。

来源顺序由 `source_priority` 配置，并轮转已知相机/序列组；来源优先不是逐图清晰度评分。LILA 任一框面积比例严格大于1/16即可，全部框保留；负样本不受此限制。调整抽样策略不需要重建索引，但必须使用新输出目录重新抽取。

`preview` 和 `sample` 都支持 `--lila-min-area 1/16`，兼容别名 `--lila-ratio`、`--lila_ratio`。参数覆盖 policy 中的 `lila_min_area`；不传则使用 policy 值（缺省1/16）。支持小数或分数，范围为 `[0,1)`。这是**动物框面积/整图面积门槛，不是 LILA 来源图片的占比上限**；设为0即不排除任何合法正面积框。预览和正式抽样应使用相同参数，两份报告都会列出实际阈值。调整该门槛不需要重建索引。

```bash
python animal_data.py sample \
  --index datasets/animal_index.sqlite \
  --policy configs/camping19.yaml \
  --lila-ratio 1/16 \
  --max-images 30000 --output yolo_datasets/animals_30k_19
```

沿用含不支持类别则整图排除的策略，不偷偷删除正框。已知同文件/硬链接重复只占一个采样槽；同文件标签冲突则全部排除并报告。先冻结全池相机/序列/文件连通组，再采样，最后按80%/10%/10%启发式分组划分。缺少LILA分组的记录保守留在训练集；比例和每类覆盖可能无法精确满足。独立复制的图片、未知相邻帧未做内容去重。

写出前验证源标注指纹和选中图片文件元数据，从保存的原像素框或原YOLO行重新计算核对索引坐标；写出后逐项核对标签ID/坐标原文和硬链接身份。证明的是转换与路径对应，不证明源标注语义或视觉质量绝对正确。未完成输出带 INCOMPLETE.txt，不用于训练。最终 REPORT.md 列出ID、各集合图片数、原类别/来源采样情况和审计数量，report.json 保存配置快照与机器明细。

## 调整方案与通用输入

当前19类为熊、野猪、蛇、狼、虎、豹、狮子、大象、鹰、鳄鱼、狗、猫、牛、羊、马、猴、鹿、鸟、小型兽类。小型兽类使用 `member_tag: small_mammal` 自动展开通用目录中已确认的小型兽类成员及下级，包括獴、鼬、灵猫、貂、水貂、兔、鼠、獾、豪猪、树鼩、河狸等，不限于狐狸或固定13种。松鼠、蝙蝠、穿山甲按当前方案显式排除；没有类别名的泛化“未知小动物”不冒充具体成员。

复制 policy 即可定义新方案：

```yaml
groups:
  large_animal:
    member_tag: large_mammal
  small_animal:
    members: [fox, rabbit, mongoose, weasel, civet]
source_priority: [ap10k, animals, lila]
negative_fraction: 0.05
val_fraction: 0.1
test_fraction: 0.1
seed: 42
```

目录中的体型标签是人工审核的业务分组，不按框面积猜动物体型。新增标签只需编辑目录；物种归一关系可通过 source_names 按数据集单独指定。父类包含所有后代，配置用 exclude 排除子类，例如 bird 排除 eagle；分组重叠、未知成员和层级循环均报错。

其他YOLO/COCO数据用 `--sources sources.yaml`，路径相对清单文件：

```yaml
sources:
  - name: my_yolo
    family: animals
    kind: yolo
    yaml: /path/to/data.yaml
  - name: my_coco
    family: lila
    kind: coco
    catalog_scope: swg-camera-traps
    images: /path/to/images
    annotations: [/path/to/annotations.json]
    require_groups: true
```

COCO bbox 必须是像素xywh，图像需有width/height。没有框不自动视作负样本：仅接受 LILA `selection_role: negative` 或清单显式 `empty_images_are_negative: true`（只用于已确认完整标注的数据）。相机/序列支持 camera_id/location/sequence_id/seq_id/video_id；其他字段需要适配，不能保证所有未知元数据都自动识别。新增来源可与三个快捷来源同时使用，name必须唯一；修改政策无需重新索引。

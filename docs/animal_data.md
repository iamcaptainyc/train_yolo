# 通用动物索引与逐层均衡抽样

新入口 `animal_data.py` 独立于旧版固定56类的 `extract_camping.py`。索引保存原始类别和框，`animal_catalog.json` 保存通用层级与体型标签，`camping19.yaml` 只是可替换的输出方案。修改分组/标签不重建原始索引，不强制19类或30K上限。

## 1. 建立索引

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

按最终类→直接成员→成员的细类逐层轮转均衡；例如獴有10个细类、鼬只有2个，先给獴/鼬相同成员配额，再在各自内部平分。粗标注占独立的原始标签桶，不能假装知道具体物种。稀缺成员取全后其他成员补足。多类别同图只存一次并同步更新所有相关成员计数，因此计数可重叠、配额为软目标，总图片数为硬上限。

同一细类中再按 AP-10K→Animals→LILA 的来源优先级抽取，并轮转已知相机/序列组。来源优先不是清晰度评分。LILA 任一框面积比例严格大于1/16即可，全部框保留；负样本不受此限制。

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

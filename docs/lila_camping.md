# LILA 露营动物类别：常用细类与大类共存

这一阶段只处理五个 LILA 数据集，不合并 `animals-detection-images-dataset`。使用固定映射 [configs/lila_camping.json](../configs/lila_camping.json)，覆盖五个原始 bbox 文件的全部 258 条类别记录。映射按数据集和原始名称匹配，原 ID 留作追溯，不跨文件混用 ID。

## 已经有通过审计的 LILA 输出：使用这个命令

在服务器仓库目录执行，`--remap-from` 填上次生成 `AUDIT_REPORT.md`、`report.json`、`data.yaml` 的输出目录：

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt

python prepare_lila_audited.py \
  --remap-from yolo_datasets/lila \
  --output yolo_datasets/lila_camping \
  --workers 16
```

该命令自动启用 `--category-policy camping`，从旧断点恢复原始下载目录、数据集列表、相机分组配置、seed 和划分比例，强制使用硬链接。新输出目录须与旧输出及原始下载目录分开，不能嵌套。

**不会运行全量 `Validate and hash`，不会重新解码或计算图片内容哈希，不会复制一份图片，不会重新划分 train/val/test，也不会改写旧输出。** 会读取原始标注 JSON、核对批次文件属性和旧 YOLO 标签，逐框生成并独立核验新标签；生成预览时只解码抽样图片。这些元数据和标签的遍历仍需要时间，进度条分别显示。

硬链接只新增目录项，不新增一份图片数据。新 `.txt`、Markdown、类别映射、少量预览和批次进度会占空间。硬链接共享图片内容，勿在任一目录修改图片。

Ctrl+C 后重新执行同一命令即可。已存在且正确的图片硬链接、标签会复用，校验与报告汇总会重新扫描元数据，不会进入图片全量校验。中断时没有正式 `data.yaml`，成功完成才生成。

快速模式要求旧任务已完成审计、保有轻量批次断点、原始下载图片/筛选标注未变化，且旧输出图片与原图为硬链接。源文件属性、筛选标注、旧标签、旧输出数量或划分证明异常时会明确停止，**绝不静默回退到全量图片校验**。如果旧输出尚未审计成功，先按原命令加 `--resume-output` 完成旧审计。

继承的是旧图片校验和精确重复图片分组证明。相机、序列交集及全部新标签重新核验；不声称重新计算唯一图片哈希数量，也不声称重新排查不同文件之间在新类别下的重复内容标签冲突。正常图片不保存逐图校验记录，沿用原来的批次断点机制。

恢复细类后，如果某细类只出现在 val/test、训练样本为零，会自动整图排除该类涉及的图片，并在报告列出 `untrained_classes_excluded`，不挪动相机组，也不隐去框作为负样本。这样正式输出的每个类别都有训练样本。

旧规则下未验证或已剔除的图片不会自动加回。若想扩展到这些图片，须选择下面的原始输入路径，进行必要的首次图片校验。

## 尚未转换过的原始 LILA 下载目录

```bash
python prepare_lila_audited.py \
  --source datasets/lila \
  --output yolo_datasets/lila_camping \
  --category-policy camping \
  --copy-mode hardlink --workers 16 \
  --val-fraction 0.05 --test-fraction 0.05
```

这条命令是首次转换，会完整检查图片。按相机/序列/相同内容连通组，以 90%/5%/5% 为目标划分，不能拆组凑比例。快速重映射则保留原来的分组和划分，实际比例见报告。由于新类别不再跟随祖先上卷，保留细类不代表每个细类都一定在 val/test 出现，报告列出各划分缺席类别。

## 合并规则

- 输出使用英文 `snake_case`，类别 ID 只为实际入选有框类别分配。没有样本的 `wolf`、当前零框的 `snake` 等不创建空类别；本阶段不会创建另一个 80 类数据集独有的类别。
- 熟悉的细类与大类共存，如 `bear` / `black_bear`、`fox` / `red_fox`、`bird` / `crow` / `woodpecker`、`rodent` / `mouse` / `squirrel`。一个框只有一个训练类别；父类关系仅供展示与后续推理，不产生额外重复标注。
- 亚洲黑熊和美洲黑熊合为 `black_bear`；马来熊和未识别熊归 `bear`。类别名称代表产品常用类别，不保证每个类别都是单一生物种，事件总结不能把“黑熊”擅自写成“亚洲黑熊”。
- `eurasian_wild_pig` 和 `bushpig` 统一为 `wild_boar`，是用户约定的野生猪类应用分组；原始物种名称在报告保留。原始 `pig` 保留为 `pig`，不因旧科学映射指向野猪就改变语义；疣猪仍为 `warthog`。当前 Caltech 的 `pig` 原始框为零，因此这次 LILA 输出可能没有 `pig` ID，后续合并另一个数据集时再启用。
- 未识别动物、未识别哺乳动物、未识别小型动物、`other_animal`、`small_mammal` 等整图排除，不输出 `animal` / `mammal` / `small_mammal`。未识别熊、狐、松鼠、鸟等仍有明确动物分组的标签可映射到该分组，不能把所有带 `unidentified` 的标签一律删掉。
- 未识别有蹄类、爬行动物或两栖动物、无脊椎动物这类无法落入已约定类别的混合粗标签，也明确排除并记录原因；不把它们猜成鹿、蜥蜴或昆虫。所有含排除类别的整图都排除，不能删除框后充当负样本。
- 猪、狐、鼷鹿、竹鼠等按原英文标签的语义处理，不用旧官方科学分类映射替它们猜物种。竹鼠使用 `bamboo_rat`；鼷鹿使用 `chevrotain`，不能并成羚羊。
- 补充此前类别概览漏列的 `tree_shrew`（树鼩）、`chevrotain`（鼷鹿）、`galago`（婴猴）、`buffalo`（非洲水牛）和 `hyena`（鬣狗）。它们不能准确改称鼠、羚羊、猴、家牛或犬。
- `civet` 是通俗灵猫类应用分组，收纳椰子狸、果子狸、非洲灵猫、獛、林狸和熊狸等；`mustelid` 为鼬科宽标签及本版不细分的貂类。它们是英语常用分组，不是拉丁学名。
- 人和车辆有明确类别映射，但当前下载筛选流程排除了其框，通常不会生成对应 ID。空图沿用有明确 empty 证据的已选负样本，不从缺框推断为空图。
- 原始标注、数据集、图像 ID、原框、原始名称和最终英文类的对应在报告中保留。模型不会因为英文名有父子关系就自动学会不确定时回退；这部分需要后续推理策略和验证。

完整原始类别去向见[258 条原始类别映射概览](lila_camping_mapping.md)；运行后的真实数量以输出报告为准。

## 输出

```text
yolo_datasets/lila_camping/
  data.yaml
  images/{train,val,test}/...
  labels/{train,val,test}/...
  AUDIT_REPORT.md
  report.json
  class_mapping.json
  taxonomy_decisions.json
  class_hierarchy.json
  audit_sources.json
  excluded_images.jsonl
  validation_errors.jsonl
  repairs.jsonl
  previews/...
  .lila-state/...
```

`AUDIT_REPORT.md` 包含原 metadata/bbox 全部类别及中文对照、原始/筛选/保留框数、每个来源类的合并与排除原因、最终英文类的中文含义与展示分组、train/val/test 数量、缺席类、框图及反算坐标误差。旧科学分类模式仍可使用，但露营任务请按本页命令显式选择新策略。

训练使用新 YAML：

```bash
python train.py --data yolo_datasets/lila_camping/data.yaml \
  --model weights/yolo11s.pt --epochs 100 --batch 128 --workers 16 --device 0
```

后续合并另一个 YOLO 数据集时须按英文语义重排 ID，不能拼接原来的数字 ID，也不能沿用旧 61 类模型的类别解释直接推理新 YAML。

现在可使用 [merge_yolo_datasets.py 合并说明](merge_yolo.md) 完成与 Animals Detection 的合并。该合并步骤只依赖处理后的 `lila_camping`，不依赖旧的 `yolo_datasets/lila`；旧目录的删除条件和硬链接空间说明见该文档。

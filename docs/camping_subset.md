# 提取最多 30000 张的露营动物数据集

入口 `extract_camping.py`：直接读取之前处理好的 LILA Camping 与 Animals Detection 两个 YOLO 数据集，按已确认的 [82 类规则](../configs/camping82.json) 合并标签、均衡选图并生成独立小数据集。无需先重新生成完整大合并集。

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt

python extract_camping.py \
  --lila yolo_datasets/lila_camping \
  --animals yolo_datasets/animals-detection-images-dataset \
  --lila-source datasets/lila \
  --workers 16
```

默认输出：`yolo_datasets/camping82_30k/`。若这些路径没变，也可直接 `python extract_camping.py --workers 16`。

## 选图规则

- **总量严格不超过 30,000 张**，包含负样本。`--max-images` 可改小，不能改到超过 30000。
- **每个类别、每个目标划分内，先用 Animals 的候选，再用 LILA 补充。** 不是把 Animals 所有图片无条件全收；仍要满足类别范围、全图标签有效及均衡预算。LILA 在相机/序列连通组间轮流取样，减少从单一相机连续抽图。
- 来源优先不等于清晰度认证：不重新解码/打清晰度分数，源数据应先完成之前的图片清理。有清晰度差异的 LILA 图仍可能入选。
- 每轮优先补当前图片数最少的类别。默认 `--class-target 1200` 为每类软目标，在各划分按比例分配；不是强制每类 1200 张，更不是把少样本改叫 mammal/rodent。
- 少样本类别尽量保留；不复制或过采样图片凑数量。多类图片整体入选，保留所有有效框，因此共现可能让某类超过软目标。**总图片上限始终不突破。**
- 默认负样本预算最多为总上限的 5%。只使用原本有空标签的图；缺标签、删掉所有范围外框的图不能算负样本。可用 `--negative-fraction 0` 禁用负样本。
- 不能兼顾分组、软目标和样本来源时，允许少于 30000 张，不靠重复图填满。报告给出实际总量、配额和欠缺类别。

## 类别

固定候选清单 82 类：此前 80 类，加上单独的黑熊 `black_bear` 和棕熊 `brown_bear`。北极熊、大熊猫仍独立，`bear` 保留给其他/未明确的熊，不能理解为无威胁。没有可用样本的候选不写入训练 names，实际类别数可能小于 82，报告会显示所有缺失项。

- 狼/狗/狐狸分开，猪/野猪分开；狼沿用你之前的胡狼、郊狼应用分组，不代表已经获得灰狼样本。
- 兔、鼠、仓鼠、松鼠、猫、牛、绵羊、山羊等保留各自名称；不做少样本上卷。
- 鹿、羚羊、猴、松鼠等按审核过的层级向上合并，不拆过细物种。
- 蝴蝶、蛾、毛毛虫、瓢虫等昆虫统一为 insect；蜘蛛、蝎子、蜈蚣各自保留；蜱虫不纳入。
- 普通鱼与金鱼合并，鲨鱼独立；海豹与海狮分开；海龟陆龟并为 turtle。
- rodent/mustelid/wild_cat 等无法安全确定到约定类别的原始粗标签不猜成鼠/鼬/家猫；**含范围外或无法安全映射框的图片整图排除**。马来熊合入 bear。懒熊没有纳入本次确认的 82 类，若遇到明确标签排除，不并入“低风险熊”。

LILA 原标注通过框坐标匹配恢复类别，保留原坐标文本；无法唯一匹配时保留输入粗类，再根据范围规则决定纳入或排除。报告不将无法恢复的细类说成已识别。

## 划分和防泄漏

默认目标 train/val/test=90%/5%/5%。先在完整候选池中将相机、序列、已知同图身份绑定成连通组并分配集合，再在各集合抽样；不会把同组样本分到不同集合来凑比例。

完整相机里可以只选部分图片，但该相机其余被选中的图片仍只能在同一集合。某些集合的可用样本少时，预算可回分配到其他集合，最终比例以报告为准。少数类未必有足够独立分组出现在全部三个集合。

LILA 分组来自 `datasets/lila/<子数据集>/selection/selected_annotations.json`；缺失时保守将该子数据集限于训练集。Animals 没有相机/序列元数据时按单图分组，不能保证未知近似图完全隔离。已有外部分组可传 `--group-map`，格式沿用 [合并说明](merge_yolo.md)。

默认不读取图片字节算哈希，仅以文件身份判断已知重复；同一硬链接文件只抽一份，优先保留 Animals 的记录。独立复制的相同图和近似图没有被全量排查。

## 存储、恢复和报告

图片继续用硬链接，不复制、不缩放、不重编码；标签是独立新文件，只改类别 ID，合并同类同框重复记录。输入目录保持不变，跨磁盘无法创建硬链接时明确报错。

Ctrl+C 后重跑同一命令：重新扫描标签和分组标注，不执行 Validate and hash、不解码图片，复用已写好的匹配硬链接及标签。只保存整体配置/输入摘要，不保存正常图片逐图校验缓存。修改类别规则、预算、seed 或输入后需换一个输出目录，防止混入旧文件。

只有完整核对硬链接身份和坐标文本成功，才生成 `data.yaml`：

```text
yolo_datasets/camping82_30k/
  data.yaml
  images/{train,val,test}/{animals,lila}/...
  labels/{train,val,test}/{animals,lila}/...
  SUBSET_REPORT.md          # 中文报告：82 类是否有样本、每类两来源数量、划分和映射
  MERGE_REPORT.md           # 同一份报告的兼容文件名
  report.json
  class_mapping.json
  class_hierarchy.json
  excluded_images.jsonl    # 仅范围排除/异常；正常未抽中只汇总计数
  .merge-state/run.json
```

先查看 `SUBSET_REPORT.md`，确认实际类别、每类样本量及缺失集合，然后训练：

```bash
python tune.py --data yolo_datasets/camping82_30k/data.yaml --batch 128 --workers 4 --trials 1
```

本地验证使用小型 YOLO 目录、硬链接和模拟图片字节检查提取逻辑；未在服务器的完整数据集上代跑，实际入选数量以远端报告为准。

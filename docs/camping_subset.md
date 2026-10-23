# 从三个来源提取最多 30000 张露营动物图片

入口：`extract_camping.py`。输入为原始 AP-10K，以及之前转换好的 LILA Camping 和 Animals Detection。

处理顺序：**合并 AP-10K 一套标注和另外两个来源 → 统一 81 类 → LILA 面积过滤 → 全局均衡抽样 → 分组划分 train/val/test → 写 YOLO 和报告**。优先级为 **AP-10K → Animals Detection → LILA**，每类尽量先用前两个来源。

## 命令

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt

python extract_camping.py \
  --ap10k /你的数据路径/imlti/ap-10k/ap-10k \
  --lila yolo_datasets/lila_camping \
  --animals yolo_datasets/animals-detection-images-dataset \
  --lila-source datasets/lila \
  --lila-min-area 1/16 \
  --workers 16
```

`--ap10k` 指向含有 `annotations/` 和 `data/` 的目录；也支持再外面一层 `ap-10k/`。其余源路径与之前相同可省略：

```bash
python extract_camping.py --ap10k /你的数据路径/imlti/ap-10k/ap-10k --workers 16
```

默认输出 `yolo_datasets/camping81_ap10k_30k/`。自定义名字用 `--output yolo_datasets/你的名字`。旧的两来源抽样检查点与新版不兼容，使用新输出目录，源数据不删除。

## AP-10K 读取方式

- 直接读取 COCO JSON 的 images、annotations、categories，根据原图 width/height 将 bbox `[x,y,w,h]` 转为 YOLO 归一化中心点和宽高，保留 10 位小数；不需要姿态关键点，不打开图片解码。
- 默认读取 `ap10k-train-split1.json`、`ap10k-val-split1.json`、`ap10k-test-split1.json`，先汇总为一个候选池，再抽样和重新划分。不会保留原 AP-10K 划分。
- split1/2/3 是不同划分方案，**只选一套**，不会将九个 JSON 全部拼起来造成重复。可用 `--ap10k-split 2` 或 `3` 切换。
- 不建立一份全量 AP-10K 图片副本或中间 YOLO 数据集。只对最终选中的图片写硬链接和 YOLO 标签。
- 超界/无效 bbox、crowd/ignore、未知类别、缺图等记录为问题，不静默裁框或把缺标图片当负样本。没有动物标注的 AP-10K 图片不自动当作可信负样本。

## LILA 框面积门槛

面积占比 = **归一化框宽 × 归一化框高**，不需要读取图片：

```text
任意一个动物框面积 / 整图面积 > 1/16，保留整张图片及全部框。
所有框都 <= 1/16，整图排除。
```

1/16=6.25%，是面积比例，不是边长比例。等于阈值也不入选。大框达标后，其他较小的动物框全部保留，避免制造漏标；此门槛只作用于 LILA，AP-10K 和 Animals 不受限制。

默认 `--lila-min-area 1/16`，也支持 `0.0625`；例如 `--lila-min-area 1/8` 更严格，`--lila-min-area 0` 允许所有有效正框。原本有空标签的负样本没有框，仍按独立负样本预算抽取，不参与这个面积判断。

面积不是清晰度分数。大目标也可能模糊，脚本不声称已评估画面质量。

## 81 类映射

固定候选清单见 [camping82.json](../configs/camping82.json)。无有效样本的类别不分配训练 ID，报告会列出缺失项。

- 熊、黑熊、棕熊、北极熊、大熊猫分开。bear 表示其他/未明确的熊，不代表无威胁；马来熊合入 bear。懒熊不在本次 81 类内，遇到明确标注排除。
- 狼、狗、狐狸分开；wolf 包含 AP-10K 原始 wolf 与原 LILA 胡狼/郊狼业务分组，不能把所有样本看成纯灰狼。猪和野猪分开。
- 兔、鼠、仓鼠、松鼠、猫、牛、绵羊、山羊等保持常见名称，不做少样本上卷。
- AP-10K 的 argali sheep→sheep、arctic fox→fox、moose→deer、bobcat→lynx、king cheetah→cheetah、snow leopard→leopard、mouse/rat→mouse、hippo→hippopotamus、rhino→rhinoceros。属于本项目的通俗上级分组，不把它们都说成同一生物物种。
- AP-10K 的 bison、panther、beaver、mole、chimpanzee、gorilla、orangutan 没有本次安全对应的类别，整图排除，不猜成家牛、豹、鼠或猴。完整原名映射见 report.json 的 ap10k 字段及 class_mapping.json。
- 鹿、羚羊、猴、松鼠等合并审核过的细类；昆虫统一 insect，蜘蛛/蝎子/蜈蚣独立，蜱虫不纳入。
- rodent/mustelid/wild_cat 等无法安全对应的粗标签不猜成鼠/鼬/家猫。**任何来源含范围外或无法安全映射的框，整图排除，不单独删掉该框。**

LILA 仍读取原始 selected_annotations.json，通过框坐标匹配恢复作者类别，再做归并。无法唯一恢复的保留输入粗类并报告，不根据照片猜物种。

## 抽样、划分、来源优先

- 总数严格不超过 `--max-images`（默认 30000，不能超过 30000），包括负样本；不复制图片凑数量。
- 在全局类别配额内，先抽 AP-10K，AP 候选不足再抽 Animals，最后才补 LILA。不把原 train/val/test 当作配额约束。
- 默认每类软目标 `--class-target 1200`，优先补图片数最少的类别。少样本尽量用完；多类图片保留全部框，共现可能让个别类别超过软目标，但总数上限不会突破。
- 默认负样本预算最多为总上限的 5%，仅取原本空标签的图。可用 `--negative-fraction 0` 禁用。
- 抽样前绑定完整池中的相机、序列和已知相同文件身份，保存连通关系；即使连接两个组的中间图片未被抽中，关联也不会丢失。抽样后才将保留组分配到约 80%/10%/10%。
- 不拆相机/序列凑比例。LILA 缺分组的子数据集保守放训练集；AP-10K/Animals 若无序列元数据按单图分组，因此不能证明未知近似图完全隔离。可用 `--group-map` 补充已知分组，格式参照 [合并说明](merge_yolo.md)。
- 只用设备/inode 检查已知重复并保留一份，优先 AP-10K、其次 Animals；不全量读图片内容算哈希，独立复制的重复图和视觉近似图未被排查。

## 输出与续跑

图片只用硬链接，原目录不改动；跨磁盘无法链接时明确报错。已有 YOLO 框坐标 token 原样保留；AP-10K 框按源 COCO 数值转换，并在输出审计中重新计算核对。只有检查通过才发出训练 data.yaml。

```text
yolo_datasets/camping81_ap10k_30k/
  data.yaml
  images/{train,val,test}/{ap10k,animals,lila}/...
  labels/{train,val,test}/{ap10k,animals,lila}/...
  SUBSET_REPORT.md          # 各类三来源数量、划分、缺失类、面积门槛及排除统计
  MERGE_REPORT.md           # 同一报告的兼容名称
  report.json
  class_mapping.json
  class_hierarchy.json
  excluded_images.jsonl    # 仅问题/范围外明细，正常未抽中只汇总计数
  .merge-state/run.json
```

Ctrl+C 后用同一命令恢复：重扫标签/标注元数据，不进行 Validate and hash，不解码图片，复用已写好且一致的硬链接和标签。修改来源、AP 划分方案、面积阈值、预算或规则时换新输出目录。统计/检查点不保存每张正常图片的校验记录。

完成后先看 SUBSET_REPORT.md，再训练：

```bash
python tune.py --data yolo_datasets/camping81_ap10k_30k/data.yaml --batch 128 --workers 4 --trials 1
```

本地测试验证 COCO 转换、严格面积边界、任意框达标保留所有框、来源优先、分组桥接、硬链接及断点复用。完整数据集需在服务器实际执行，最终数量和类别以报告为准。

鬣狗（hyena）排除，含鬣狗的混合图片整图排除。按最终抽中的每类图片数优化 8:1:1，同图同类多个框只计一张；保留相机/序列/已知重复图分组，比例可能受大组限制。报告列出实际比例及偏差。

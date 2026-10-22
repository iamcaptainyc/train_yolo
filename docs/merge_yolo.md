# 合并 LILA Camping 与 Animals Detection YOLO 数据集

入口：[merge_yolo_datasets.py](../merge_yolo_datasets.py)。输入是两个已转换好的 YOLO 数据集目录或 `data.yaml`；LILA 必须先按 [露营类别方案](lila_camping.md) 处理为英文常用类别，不使用旧科学分类的 `yolo_datasets/lila` 作为合并输入。

## 直接运行

在服务器仓库目录执行：

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt

python merge_yolo_datasets.py \
  --lila yolo_datasets/lila_camping \
  --animals yolo_datasets/animals-detection-images-dataset \
  --lila-source datasets/lila \
  --output yolo_datasets/camping_all \
  --min-class-images 100 \
  --max-class-images 10000 \
  --cap-class-images 0 \
  --workers 16
```

也可以给 `--lila`、`--animals` 传各自的 `data.yaml`。脚本支持本项目的目录式 YOLO 划分，例如 `images/train`、`images/val`、`images/test`，对应 `labels/`；不支持 URL、路径列表、txt 划分清单或分割/关键点标签。

输入只读。输出必须是独立的新目录，与两个源目录不能重叠或嵌套。图片只使用 `os.link` 硬链接，不缩放、不重编码、不重新解码，也没有复制回退；跨磁盘不能硬链接时会明确失败。每张图加来源命名空间和稳定文件名，两来源同名图片不会覆盖。标签生成新文件，不能硬链接修改源标签。

## 类别与划分

- 按英文语义统一为 `snake_case`，重新生成连续 ID；不拼接原始数字 ID。
- `bull`、`cow`、`cattle`、`domestic cattle` → `cattle`。
- `brown bear` → `brown_bear`，`polar bear` → `polar_bear`；`bear` 与 `black_bear` 等细类允许共存，不整体上卷。
- `wild pig`、`wild_boar`、`bushpig`、`eurasian wild pig` → `wild_boar`；`pig` 保持独立，不能从该名称推断一定是家猪；`warthog` 仍保留。
- `moths and butterflies` → `insect`，`butterfly` 仍保留；不能把混合蛾蝶标签全部猜成蝴蝶。
- `domestic dog` → `dog`；`domestic cat` → `cat`；`mountain lion` → `cougar`。
- `jackal`（胡狼）与 `coyote`（郊狼）统一为用户定义的 `wolf`，中文“狼类（含胡狼、郊狼）”。这是应用分组，不是灰狼物种标注，也不意味着新增了灰狼样本。统一在少样本阈值统计之前进行，报告保留各来源名称和 ID。
- 未确定动物、哺乳动物、小型动物等宽泛类别整图排除。未审阅的新名称或旧科学分类会明确报错，不猜物种。类别仅为实际保留的有框样本分配 ID，缺失的狼等不会凭空加入。
- 默认重新划分，不保留原 train/val/test。目标为 90%/5%/5%，通过 `--val-fraction .05 --test-fraction .05 --seed 42` 调整。完整相机/序列连通组不可拆；先争取少数类覆盖，再优化图片比例和类别分布，不保证精确比例或找到所有可行覆盖方案。原来只在 val/test 的类别也可以重新分配到 train。
- 默认 `--min-class-images 100`：按图片数统计，同图同类多个框只计一张。少于 100 张的细类归入合理上级；上级支持量包含子类覆盖图片，避免因细分而错误上卷。`0` 禁用少样本合并。旧参数 `--min-class-boxes` 是兼容拼写，也按图片数计。
- 例如竹鼠→`rodent`，巨蜥→`lizard`，水貂/鼬→`mustelid`，少量鸽子→`bird`，海马→`fish`，豹猫→`wild_cat`。无合理上级的蝙蝠、穿山甲等保持原类，在报告列出，不能错并为鼠或重新引入未知动物类别。`wild_boar`、`pig`、`warthog` 强制保持独立。提高阈值也可能合并棕熊等熟悉细类，应查看报告再训练。
- 每个保留框的四个坐标 token 原样保留，包括小数位与科学计数法；只改变类别 ID。若别名合并造成同类同坐标的重复记录，保留第一条原始坐标文本，报告去重数量。
- 既有空标签作为负样本保留。缺失标签、非法 ID、非法坐标或越界框会整图排除，绝不把丢失正标签当作负样本。继承两个来源既有的负样本，总负样本比例不重新抽成 5%。

## 超过 10,000 张优先细分，保留全部有效图片

`--max-class-images 10000` 在图片数严格大于 10,000 时触发。默认从 [细分规则](../configs/lila_refinements.json) 中的 74 条数据集专属映射恢复英文细类，例如白尾鹿、黑尾鹿、水鹿、不同松鼠、羚羊和鸟类。不把 `unidentified` 或仅有宽泛标签的图片猜成某个物种；`wild_boar` 继续统一，不恢复成多种野猪标签。

细类恢复读取原始 selected_annotations.json，使用原图尺寸与原始框重算规范化坐标，与当前 YOLO 框逐一匹配（沿用 8 位小数精度）。匹配唯一才修改类别，保留现有坐标文本；缺尺寸、坐标不匹配、同框原类别有歧义时保留粗类并汇总原因。不重新解码图片，不全量读取图片内容，不生成逐图校验缓存。Animals 原始 YOLO 中没有更细的来源信息，保持其既有标注粒度。

旧 LILA 策略错误地将黄喉貂归到了 bird；本次在处理超量 bird 时，对原始黄喉貂框匹配成功的记录改为黄喉貂，再按数量决定是否归入 mustelid。无法验证的记录保留并报告，不假称已修复全部源标注。

**默认 `--cap-class-images 0`，不按类别数量抽样。** 10,000 张仅为细分触发阈值，不是上限。已是单一物种、缺少更细原标注或无法可靠恢复细类时，即使超过 10,000 张也保留全部有效图片，只在报告列出。少样本合并只改类别，不减少图片。

旧版命令中的 `--cap-class-images 10000` 会主动抽样，必须删除该参数或改为 `0`。正在运行旧命令可 Ctrl+C，更新代码后使用新输出目录 `camping_all` 重跑；输入图片没有被删除，新的输出能重新纳入之前仅因数量被抽样排除的图片。无需重新解码或全量哈希。

仅显式传入大于零的 `--cap-class-images` 才启用旧的整图抽样功能。缺失/非法标签、已发现的冲突等质量排除规则仍保留，完整原因见报告；不承诺无效输入也进入训练集。

统计中的图片数是输入图片记录数，不是内容去重后的唯一图片数。多类图片计入多个类别，各类图片数之和可以超过总图片数。最终类别数和图片分布须以实际运行报告为准，不能用旧框数表直接推算。

## 重复图片与检查范围

默认不读图片主体字节、不计算图片内容哈希，只检查设备号/inode 是否指向同一个文件。原划分之间的同图绑定为同一连通组再分配；统一标签冲突时相关记录全部排除。同图同标签的记录保留，图片数按来源记录计数，不声称是唯一内容图片数。

LILA 从 `lila_camping/.lila-state/run.json` 恢复原下载目录，再读取各 `selection/selected_annotations.json` 的相机/序列字段；原下载目录移动后用 `--lila-source` 指定。只读取标注，不需要原始下载图片主体，不运行 Validate and hash。相同相机、跨相机同序列、已知同图通过连通关系绑定。某个 LILA 子数据集缺失分组信息时，保守将该子数据集放在训练集并在报告提示，不随机拆图。

Animals 默认按单图分组，因为当前 YOLO 输入没有相机/序列元数据；不能声称其未知序列或近似图片已隔离。若有分组信息可传 `--group-map groups.json`，格式为 `{"animals/train/relative/image.jpg": "sequence-001"}`；键是来源/原划分/相对图片路径，值为全局分组 ID，相同值强制绑定。该参数也可以补充 LILA 分组，但不解除从官方元数据得到的相机约束。

`--preserve-splits --min-class-images 0 --max-class-images 0 --cap-class-images 0` 可使用保留旧划分且不合并稀有类的模式，此模式跨原划分同图才会全部排除。新策略与旧输出检查点不兼容，必须使用新的输出目录；脚本不会自动删除旧数据。

**默认模式不能发现所有独立复制文件的内容重复**，也不声称不存在全部跨数据集内容泄漏。需要检查字节相同的独立文件时，首次运行就添加：

```bash
python merge_yolo_datasets.py \
  --lila yolo_datasets/lila_camping \
  --animals yolo_datasets/animals-detection-images-dataset \
  --output yolo_datasets/camping_merged_content_checked \
  --workers 16 --check-content-hashes
```

该选项仅对大小相同的候选文件计算 SHA-256，唯一大小文件无需读取；同 inode 的候选只读取一次。同字节文件绑定为同一连通组。不会解码图片，也不会保存正常图片的逐图哈希记录。大部分图片大小重复时，仍可能读取很多图片，需要时间；中断后哈希在内存中重新计算。它只能发现字节相同的文件，不能发现不同编码或相邻帧的视觉近似图。重新划分不继承旧划分的内容哈希隔离证明，默认仅保证本次恢复的相机/序列与已知文件身份分组隔离。

两种模式都会重新读取实际输出标签，核对统一类别和原坐标文本，并逐图核对硬链接文件身份。**不重新认证源图片是否损坏、作者物种是否标对、框是否标全**；输入应先完成各自的转换检查。

## 中断与输出

Ctrl+C 后重跑完全相同的命令，已写好且正确的硬链接和标签会复用；重新扫描标签与元数据，不保存正常图片的逐图校验表。正常状态只保存 `.merge-state/run.json` 里的配置和一个整体输入摘要。输入配置、类别规则、保留样本或文件属性变化时须选新输出目录，避免混合版本；线程数可改变。

`Hardlink images and remap IDs` 已移除旧版本的逐图目录扫描：现在每个标签目录只扫描一次清理临时文件，目录也集中创建。相同策略中断后可复用已完成文件；从保留划分切换到重新划分，或修改阈值、种子、比例时须换新输出目录。这不会触发图片解码或默认内容哈希；速度仍受磁盘小文件创建、读取和元数据吞吐影响。

仅全部检查完成才生成正式 `data.yaml` 和 `MERGE_REPORT.md`；中断/失败时输出 `PROCESSING_REPORT.md`，保留已写文件便于续跑。

```text
yolo_datasets/camping_all/
  data.yaml
  images/{train,val,test}/{lila,animals}/...
  labels/{train,val,test}/{lila,animals}/...
  MERGE_REPORT.md
  report.json
  class_mapping.json
  class_hierarchy.json
  excluded_images.jsonl
  .merge-state/run.json
```

报告列出每个来源的原类别名/ID → 最终英文名/新 ID、少样本合并前图片数/目标、无合适上级的稀有类、每类独立分组数与缺失集合、原划分→新划分流向、各来源/各划分图片数与框数、每类 train/val/test 图片数和总框数、排除原因、坐标和硬链接检查结果。缺少独立分组时不会拆相邻帧来凑齐三集合；检查通过也不等于所有类别都有足够评估样本。文件名以 `SHA256(来源/原划分/图片相对路径)` 命名；这不是图片内容哈希。

训练使用新 YAML，旧权重的分类头 ID 不能直接解释为新类别：

```bash
python train.py --data yolo_datasets/camping_all/data.yaml \
  --model weights/yolo11s.pt --epochs 100 --batch 128 --workers 16 --device 0
```

## 旧的 yolo_datasets/lila 能否删除

**可以，但先确认 `yolo_datasets/lila_camping` 已完整重映射并审计通过，生成了正式 `data.yaml`、`report.json`、`AUDIT_REPORT.md`，且没有任务仍使用旧目录。** 本合并脚本的输入是 `lila_camping`，不依赖旧 `lila`。

删除旧 `lila` 目录只移除该目录的文件链接；原始下载目录、`lila_camping`、最终合并目录持有的硬链接仍可读取同一份图片内容。不过原重映射命令的 `--remap-from yolo_datasets/lila` 将不能再恢复，需保留旧统计报告、配置和审计来源文件以供追溯；也应保留原始下载标注，便于后续调整类别。

不要在重映射尚未成功时删除旧目录。不要把 `datasets/lila`（原始下载数据与标注）和 `yolo_datasets/lila`（旧转换输出）混淆。后者删除通常不会释放一整份图片空间，因为其他硬链接仍持有图片；主要减少旧标签、预览和目录项占用。本脚本不自动删除任何输入目录。

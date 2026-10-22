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
  --output yolo_datasets/camping_merged \
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
- 未确定动物、哺乳动物、小型动物等宽泛类别整图排除。未审阅的新名称或旧科学分类会明确报错，不猜物种。类别仅为实际保留的有框样本分配 ID，缺失的狼等不会凭空加入。
- 每张保留图片都留在来源的原始 train/val/test，绝不随机重新划分。两来源的比例不同，因此总比例不保证是 90%/5%/5%。只出现在 val/test、没有训练样本的类别，其相关图片整图排除并记录，不移入训练集，也不将物体隐去为背景。
- 每个保留框的四个坐标 token 原样保留，包括小数位与科学计数法；只改变类别 ID。若别名合并造成同类同坐标的重复记录，保留第一条原始坐标文本，报告去重数量。
- 既有空标签作为负样本保留。缺失标签、非法 ID、非法坐标或越界框会整图排除，绝不把丢失正标签当作负样本。继承两个来源既有的负样本，总负样本比例不重新抽成 5%。

## 重复图片与检查范围

默认不读图片主体字节、不计算图片内容哈希，只检查设备号/inode 是否指向同一个文件。查出跨 train/val/test 的同图时，将相关记录全部排除；同一划分里同图但统一标签冲突，也全部排除。同一划分同图同标签的记录保留，图片数按来源记录计数，不声称是唯一内容图片数。

**默认模式不能发现所有独立复制文件的内容重复**，也不声称不存在全部跨数据集内容泄漏。需要检查字节相同的独立文件时，首次运行就添加：

```bash
python merge_yolo_datasets.py \
  --lila yolo_datasets/lila_camping \
  --animals yolo_datasets/animals-detection-images-dataset \
  --output yolo_datasets/camping_merged_content_checked \
  --workers 16 --check-content-hashes
```

该选项仅对大小相同的候选文件计算 SHA-256，唯一大小文件无需读取；同 inode 的候选只读取一次。不会解码图片，也不会保存正常图片的逐图哈希记录。大部分图片大小重复时，仍可能读取很多图片，需要时间；中断后哈希在内存中重新计算。它只能发现字节相同的文件，不能发现不同编码或相邻帧的视觉近似图。源 LILA 的相机/序列隔离通过原划分继承，合并器不从图片猜新的相机信息。

两种模式都会重新读取实际输出标签，核对统一类别和原坐标文本，并逐图核对硬链接文件身份。**不重新认证源图片是否损坏、作者物种是否标对、框是否标全**；输入应先完成各自的转换检查。

## 中断与输出

Ctrl+C 后重跑完全相同的命令，已写好且正确的硬链接和标签会复用；重新扫描标签与元数据，不保存正常图片的逐图校验表。正常状态只保存 `.merge-state/run.json` 里的配置和一个整体输入摘要。输入配置、类别规则、保留样本或文件属性变化时须选新输出目录，避免混合版本；线程数可改变。

`Hardlink images and remap IDs` 已移除旧版本的逐图目录扫描：原先每处理一张图都 `glob` 整个标签目录清理临时文件，数十万张时会越跑越慢。现在只在写入前对每个标签目录扫描一次并清理本程序的临时文件，目录也集中创建。正在运行旧版本时先 Ctrl+C，`git pull --ff-only` 后重跑同一命令与输出目录，已完成文件继续复用。这不会触发图片解码或默认内容哈希；剩余速度仍受磁盘小文件创建、读取和元数据吞吐影响。

仅全部检查完成才生成正式 `data.yaml` 和 `MERGE_REPORT.md`；中断/失败时输出 `PROCESSING_REPORT.md`，保留已写文件便于续跑。

```text
yolo_datasets/camping_merged/
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

报告列出每个来源的原类别名/ID → 统一英文名/新 ID、各来源/各划分图片数与框数、每类 train/val/test 框数、排除数量及完整原因清单、坐标和硬链接检查结果、内容重复检查范围。文件名以 `SHA256(来源/train或val或test/图片相对路径)` 命名；这个文件名摘要不是图片内容哈希。

训练使用新 YAML，旧权重的分类头 ID 不能直接解释为新类别：

```bash
python train.py --data yolo_datasets/camping_merged/data.yaml \
  --model weights/yolo11s.pt --epochs 100 --batch 128 --workers 16 --device 0
```

## 旧的 yolo_datasets/lila 能否删除

**可以，但先确认 `yolo_datasets/lila_camping` 已完整重映射并审计通过，生成了正式 `data.yaml`、`report.json`、`AUDIT_REPORT.md`，且没有任务仍使用旧目录。** 本合并脚本的输入是 `lila_camping`，不依赖旧 `lila`。

删除旧 `lila` 目录只移除该目录的文件链接；原始下载目录、`lila_camping`、最终合并目录持有的硬链接仍可读取同一份图片内容。不过原重映射命令的 `--remap-from yolo_datasets/lila` 将不能再恢复，需保留旧统计报告、配置和审计来源文件以供追溯；也应保留原始下载标注，便于后续调整类别。

不要在重映射尚未成功时删除旧目录。不要把 `datasets/lila`（原始下载数据与标注）和 `yolo_datasets/lila`（旧转换输出）混淆。后者删除通常不会释放一整份图片空间，因为其他硬链接仍持有图片；主要减少旧标签、预览和目录项占用。本脚本不自动删除任何输入目录。

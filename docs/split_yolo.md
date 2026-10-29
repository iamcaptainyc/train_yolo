# 批量重新划分 YOLO 数据集

`split_yolo.py` 对每个输入数据集分别收集全部图片，打乱后重新生成 train/val/test。原来没有划分、只有 train/val、或者三份齐全，都执行重分。此步骤不合并类别，类别 ID、类别名称、标注框坐标及空标签原文保持不变。

```bash
python split_yolo.py \
  --datasets /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c \
  --output yolo_datasets/resplit \
  --ratios 8 1 1 \
  --seed 42 --workers 16
```

也可以输入单个数据集，或直接传 YAML 文件。根目录会查找 `data.yaml`、`data.yml`、`dataset.yaml`、`dataset.yml`。没有 YAML 时也能按图片和数值标签重新划分，保持原 ID、坐标及负样本；报告按实际出现的 ID 统计，名称显示“未提供名称”，不生成 data.yaml 或虚构名称。已有 YAML 则继续检查真实 `names` 类别表；已存在但内容不合法的 YAML 不会被静默忽略。

例如只有下面这几层目录，完全没有 YAML，也可以直接运行：

```text
merged6_obstacle_only/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

```bash
python split_yolo.py --in-place \
  --datasets /path/to/merged6_obstacle_only \
  --ratios 8 1 1 --workers 16
```

有 YAML 与没有 YAML 的数据集可以放在同一条 `--datasets` 命令中批量处理。没有 YAML 的来源，直接训练或按类别名称合并前，仍需提供真实 ID→名称表。

支持这些布局，子文件夹也会递归收集：

```text
dataset_a/             dataset_b/                dataset_c/
  data.yaml              data.yaml                 data.yaml
  images/*.jpg           images/train/*.jpg        train/images/*.jpg
  labels/*.txt           images/val/*.jpg          train/labels/*.txt
                         labels/train/*.txt        valid/images/*.jpg
                         labels/val/*.txt          valid/labels/*.txt
```

有 YAML 时同时读取其中有效的 train/val/test 目录、目录列表及图片清单 TXT。清单中的相对图片路径相对该 TXT 文件。原 YAML 中失效或不存在的划分路径会写入报告，仍从本地 images 目录收集文件；清单中明确列出的图片缺失则报错。缺少标签、框坐标不合法会使该数据集失败，有类别表时也检查未知 ID；无类别表时允许全部非负整数 ID。空 TXT 是合法负样本，不会把缺失标签当负样本。

## 直接在原目录修改

无需 `--output`，显式指定 `--in-place`：

```bash
python split_yolo.py \
  --in-place \
  --datasets /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c \
  --ratios 8 1 1 --seed 42 --workers 16
```

处理完成后，每份原目录变为：

```text
dataset_a/
  data.yaml                  # 直接更新 path/train/val/test
  images/train/*.jpg         # 平铺，图片内容与 inode 保持不变
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  SPLIT_REPORT.md            # 当前划分统计
  split_report.json          # 当前机器可读统计
```

先停止正在读取该数据集的训练。脚本校验图片/标签后，在原目录内暂存新的硬链接布局，再备份旧目录并替换；成功后删除临时布局和旧文件链接，不保留两套图片，也不增加图片内容占用。处理期间有临时目录 `.split_yolo_inplace/`；不要手动删它。普通失败/Ctrl+C 自动回滚；进程被强制杀死时，下次重跑同一命令会先恢复未完成的替换，再重新划分，包括 YAML 暂时位于备份中的情况。每份数据集有独占锁，防止两个脚本同时替换。

许可证等附属文件及孤立标签会保留，旧 `.cache` 清除；其他原有报告保留为历史记录，最新统计以 `SPLIT_REPORT.md` 和 `split_report.json` 为准。没有逐图片校验缓存，也不是逐图片断点续跑；中断后会重新扫描和规划。若原 YAML 名为 data.yml/dataset.yaml，原文件与 data.yaml 都会更新。

原地模式要求图片、标签位于对应数据集目录内，不处理外部路径或符号链接；多个输入不能相互嵌套。没有 YAML 时保持无 YAML 状态，只更新图片/标签布局和统计报告。有 YAML 时更新原 YAML 和 data.yaml。`--in-place` 与 `--output`、`--copy-images` 不能同时使用。一个来源失败后继续其他来源，最后控制台汇总成功/失败数量；各来源的报告直接写在各自原目录。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--datasets` | 一个或多个数据集根目录或 YAML 路径 |
| `--output` | 导出模式使用的新目录；与 `--in-place` 二选一 |
| `--in-place` | 在原目录直接重新划分，带事务回滚，不创建独立输出数据集 |
| `--ratios 8 1 1` | train/val/test 图片比例，默认 80%/10%/10%；也支持 `9 0.5 0.5` |
| `--seed 42` | 固定随机种子，输入路径和文件不变时可复现；更换 seed 可重新打乱 |
| `--workers 16` | 并发检查标签与建立硬链接，任务数量分批限制，避免一次创建大量 Future |
| `--copy-images` | 显式复制图片；跨磁盘/文件系统不能硬链接时使用 |

导出模式下图片默认硬链接到新目录，标签单独写入，源文件不移动、不删除；原地模式则直接替换原数据布局。硬链接失败不会悄悄改为复制。图片不会重新编码、解码或做全文件哈希。输出图片平铺到 images/train、images/val、images/test，文件名为来源路径的 SHA-256，避免不同目录同名覆盖。

重新划分按图片随机进行，不是按类别分层，也不读取相机/序列元数据。已知同一文件/硬链接整体分配，保留这些不同文件路径的所有图片，避免分到不同集合；因此实际比例可能偏离目标。独立复制的重复图、相邻帧不会自动识别。至少需要三个独立文件分组，无法分成三个非空集合时明确报错，不复制图片来凑数。

## 输出及后续合并

自动以来源目录名和路径短指纹命名，无需逐个指定名称：

```text
yolo_datasets/resplit/
  BATCH_REPORT.md              # 每份数据集的成功/失败、数量和输出链接
  report.json                  # 批处理结果
  dataset_a-xxxxxxxx/
    data.yaml                  # 更新后的绝对路径和 train/val/test
    images/train/              # 平铺硬链接
    images/val/
    images/test/
    labels/train/
    labels/val/
    labels/test/
    SPLIT_REPORT.md            # 每类图片数、框数、划分数量及原失效路径
    report.json
  dataset_b-xxxxxxxx/
    ...
```

导出模式下，一个数据集检查失败后继续其他数据集，最后生成批量报告，有失败时退出码为 1。未完成输出带 `INCOMPLETE.txt`，不要用于训练。中断后源数据保留，使用新的输出目录重跑；不会保存逐图片校验缓存。原地模式的恢复方式见上文。

随后将报告中显示的各输出目录传给 `merge_yolo_ids.py --datasets ...`，可继续按原 ID 筛选类别或合并。此脚本只统一目录及划分，不重新定义类别 ID；合并时仍需处理同 ID 的名称冲突及 reserved 占位定义。

# 批量重新划分 YOLO 数据集

`split_yolo.py` 对每个输入数据集分别收集全部图片，打乱后重新生成 train/val/test。原来没有划分、只有 train/val、或者三份齐全，都执行重分。此步骤不合并类别，类别 ID、类别名称、标注框坐标及空标签原文保持不变。

```bash
python split_yolo.py \
  --datasets /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c \
  --output yolo_datasets/resplit \
  --ratios 8 1 1 \
  --seed 42 --workers 16
```

也可以输入单个数据集，或直接传 YAML 文件。根目录会查找 `data.yaml`、`data.yml`、`dataset.yaml`、`dataset.yml`，必须有真实的 `names` 类别表，不猜测类别名称。

支持这些布局，子文件夹也会递归收集：

```text
dataset_a/             dataset_b/                dataset_c/
  data.yaml              data.yaml                 data.yaml
  images/*.jpg           images/train/*.jpg        train/images/*.jpg
  labels/*.txt           images/val/*.jpg          train/labels/*.txt
                         labels/train/*.txt        valid/images/*.jpg
                         labels/val/*.txt          valid/labels/*.txt
```

同时读取 YAML 中有效的 train/val/test 目录、目录列表及图片清单 TXT。清单中的相对图片路径相对该 TXT 文件。原 YAML 中失效或不存在的划分路径会写入报告，仍从本地 images 目录收集文件；清单中明确列出的图片缺失则报错。缺少标签、未知类别、框坐标不合法会使该数据集失败，空 TXT 是合法负样本，不会把缺失标签当负样本。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--datasets` | 一个或多个数据集根目录或 YAML 路径 |
| `--output` | 必须是不存在的新目录，不能位于任何输入数据集内部，也不能是其父目录 |
| `--ratios 8 1 1` | train/val/test 图片比例，默认 80%/10%/10%；也支持 `9 0.5 0.5` |
| `--seed 42` | 固定随机种子，输入路径和文件不变时可复现；更换 seed 可重新打乱 |
| `--workers 16` | 并发检查标签与建立硬链接，任务数量分批限制，避免一次创建大量 Future |
| `--copy-images` | 显式复制图片；跨磁盘/文件系统不能硬链接时使用 |

图片默认硬链接到新目录，标签单独写入，源文件不移动、不删除；硬链接失败不会悄悄改为复制。图片不会重新编码、解码或做全文件哈希。输出图片平铺到 images/train、images/val、images/test，文件名为来源路径的 SHA-256，避免不同目录同名覆盖。

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

一个数据集检查失败后继续其他数据集，最后生成批量报告，有失败时退出码为 1。未完成输出带 `INCOMPLETE.txt`，不要用于训练。中断后源数据保留，使用新的输出目录重跑；本脚本没有断点续跑。不会保存逐图片校验缓存。

随后将报告中显示的各输出目录传给 `merge_yolo_ids.py --datasets ...`，可继续按原 ID 筛选类别或合并。此脚本只统一目录及划分，不重新定义类别 ID；合并时仍需处理同 ID 的名称冲突及 reserved 占位定义。

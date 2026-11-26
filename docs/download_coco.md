# 下载 COCO 2017 动物、人员和车辆检测数据

## 统计已下载图片的同框数量

```bash
python stats_coco.py yolo_datasets/coco_animals8
```

读取本地 `_metadata/annotations_trainval2017.zip` 中的完整官方标注，统计 `images/train` 和 `images/val` 里的已有非空图片。输出 `COOCCURRENCE.md` 和 `cooccurrence.json`。每张图在每项中只计一次，分别给出人车、人与动物、动物与车、三者同框，以及互斥的单组和双组数量。三者同框也计入三种两两同框。

默认车指 `car`。可加 `--vehicle-classes car truck bus motorcycle bicycle` 扩大车的范围。官方 ZIP 在其他位置时，传 `--annotations-zip /path/to/annotations_trainval2017.zip`。统计以有效非 crowd 单体标注为准，旧 YOLO 标签即使漏了人框，也能从完整官方标注计数。下载脚本完成后也会自动输出这两份报告，其口径为本次导出的类别和框。

## 下载与追加类别

在仓库根目录运行：

```bash
python download_coco.py --classes cow elephant \
  --output yolo_datasets/coco_cow_elephant --workers 16 --retries 5
```

可选动物类别为 `bird cat dog horse sheep cow elephant bear zebra giraffe`，`cattle` 可作为 `cow` 的输入别名。也支持 `person car truck bus motorcycle bicycle`。脚本默认从 `hf-mirror.com` 下载 COCO 标注 ZIP（约 241 MiB），并通过 HTTP Range 从镜像的 `train2017.zip` 和 `val2017.zip` 中读取 ZIP 目录及命中的图片。已有标注 ZIP 可用 `--annotations-zip /path/to/annotations_trainval2017.zip` 复用。

## 给已有动物数据补充人和车

在原输出目录中追加 `person` 和 `car`，同时补齐原有动物图片里的人、车框。

```bash
python download_coco.py \
  --output yolo_datasets/coco_animals8 \
  --add-classes person car \
  --workers 16 --retries 5
```

`--add-classes` 与原类别取并集，沿用原 ID 并把新增类别追加到末尾。省略 `--classes` 时自动读取 `_metadata/classes.json`，首次创建数据集时默认选择 `cow elephant`。旧图片复用，旧标签按官方标注重新核对。狗和人同图时输出狗框和人框，人和车同图时输出人框和车框。所选类别的单体框全部保留。

如果还要覆盖 COCO 全部 10 个动物类别，可完整列出动物、人和车。已有类别 ID 仍按原顺序保留。

```bash
python download_coco.py \
  --output yolo_datasets/coco_animals8 \
  --classes bird cat dog horse sheep cow elephant bear zebra giraffe person car \
  --workers 16 --retries 5
```

需要其他车辆类别时加上 `truck bus motorcycle bicycle`。只想给已下载图片补框，可在上面的命令中加 `--local-only`。该操作使用本地官方标注 ZIP，将已经下载的匹配图及其完整目标框导出，缺失图片记录在 `_metadata/failures.json`。之后用相同命令去掉 `--local-only` 即可补图。

这些命令生成 YOLO 检测框。筛选后的 COCO JSON 也保留原标注里的 `segmentation` 字段。更新期间 `INCOMPLETE.txt` 标记本次转换尚未完成，完成后自动移除。中断后重跑会核对每份标签，恢复未补齐的框。原目录支持增加类别，缩减类别时使用新目录。

脚本校验标注 ZIP 的 SHA-256 和从图片 ZIP 中读取的成员 CRC。镜像需要支持 HTTP Range。网络中断时重跑会复用已完成的图片，未完成的图片重新读取 ZIP 对应区间。`--source official` 可使用原来的 COCO 主站和 Google Cloud 下载流程。图片下载失败时，`_metadata/failures.json` 会记录文件名和错误。

重跑同一命令会复用完整文件，并对支持 HTTP Range 的服务器续传标注 ZIP 的 `.part`。开始时按划分显示已完成、待处理和待下载数量。进度条只统计待处理项，并单独显示失败数。镜像频繁断开连接时可降低 `--workers`，继续使用同一个输出目录。每张图片保留图中所选类的有效单体框。跳过 `iscrowd` 群体框，超出图像的有效框裁到图像边界。图片保留原 `train2017` 和 `val2017` 划分。

输出含 `images/{train,val}`、`labels/{train,val}`、`data.yaml` 和各类图片数及框数 `report.json`。筛选后的官方 COCO 记录保存在 `_metadata/instances_train2017_subset.json` 和 `_metadata/instances_val2017_subset.json`，保留原始类别 ID 和坐标。

只要训练集和验证集各有一张已完成的图片和标签，下载结束后就会生成 `data.yaml`。部分下载失败时，YAML 指向只包含已完成样本的 `train.txt` 和 `val.txt`，报告记录实际数量和缺失数量。加 `--local-only` 可直接整理现有文件。补齐时去掉 `--local-only` 并重跑，继续使用原输出目录。

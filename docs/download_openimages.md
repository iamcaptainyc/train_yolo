# 下载 Open Images V7 动物图片和官方检测框

## 给已下载的图片补官方检测框

```bash
python download_openimages.py \
  --output datasets/openimages_animals \
  --annotations-only \
  --yolo-output yolo_datasets/openimages_animals_bbox
```

把 `--output` 设为原下载目录。脚本读取 `类别/train_ImageID.jpg` 等现有文件和原下载配置，下载官方检测类别表以及实际需要的分集 bbox CSV 到 `_metadata`。会复用已完成的 CSV，支持下载中断后续传。该模式无需图像级 CSV，也不会重新下载图片。训练集 bbox CSV 仍是官方整表，筛选在本地进行。

仅当图片有原目录类别的有效官方框时才导出，保留同图其他所选类别的有效框。插图框和群体框跳过。输出使用硬链接，保留 Open Images 原有分集，写入 YOLO 标签、`data.yaml` 和 `bbox_report.json`。`--classes Cat Lion Crocodile` 可只处理指定类。原图所在目录和新输出需在同一文件系统。

原先 5000 张按图像级类别筛选，与官方检测标注只有部分重合。没匹配到官方框的图片留在原目录，不作为负样本导出。官方检测类别表没有 `Wolf`，同样无法给那批狼照片直接补框。报告按类别目录统计匹配和跳过数量；最终类别 ID 以新目录的 `data.yaml` 为准。

## 按图像级类别下载照片

在仓库根目录执行：

```bash
python download_openimages.py \
  --output datasets/openimages_animals \
  --classes Wolf "Red wolf" Coyote Jackal Cat Lion Crocodile \
  --max-per-class 5000 --workers 16 --retries 5
```

默认跨 train/validation/test 汇总筛选，每个合并后的输出类别最多 5000 张。类别样本不足时全部保留；使用种子固定的蓄水池随机抽样，不要求把标签表全部载入内存。

Wolf、Red wolf、Coyote、Jackal 是原始标签中的狼、红狼、郊狼、胡狼，按业务要求统一保存在 wolf/；不是生物学分类合并。四者去重合并后 wolf 总共最多 5000 张，内部按原始类别轮流抽取，尽量均衡，某个来源不足时由其他来源补齐。Cat、Lion、Crocodile 分别保存在 cat/、lion/、crocodile/，各最多 5000 张。可用 --splits train 只筛选原训练集。

来源使用官方完整的图像级类别表和人工图像标签，仅选择 Confidence=1。图片通过官方 open-images-dataset AWS 公开桶下载，无须登录。首次下载的全类别 CSV 标签表较大，需预留额外磁盘空间和扫描时间。

完整文件复用，下载中断保留 .part 并尝试 HTTP Range 续传；服务器不支持续传则从头下载该文件。同命令可重跑，修改类别、种子、限额或分集时要求使用新的输出目录，避免把旧图片混进新统计。没有执行全图损坏检查。

输出根目录的 REPORT.md 和 report.json 统计输出类别图片数、原始类别数量和失败列表。_metadata/selection.jsonl 保存每张图片原始类别及分集。被选中且同时属于不同输出类别的图片会跳过并记录到 conflicts.jsonl；未检索类别的其他动物仍可能出现在同一图片中。

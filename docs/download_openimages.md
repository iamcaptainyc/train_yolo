# 下载 Open Images V7 动物分类图片

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

输出根目录的 REPORT.md 和 report.json 统计输出类别图片数、原始类别数量和失败列表。_metadata/selection.jsonl 保存每张图片原始类别及分集。被选中且同时属于不同输出类别的图片会跳过并记录到 conflicts.jsonl；未检索类别的其他动物仍可能出现在同一图片中，后续伪标注需要复核。

运行后可以直接生成多类别 YOLO 伪标注：

```bash
python label_md.py \
  --input datasets/openimages_animals \
  --weights weights/MDV6-yolov10-e.pt \
  --animal-class-id 0 \
  --output yolo_datasets/openimages_animals_yolo \
  --device 0 --batch 16 --imgsz 640 --conf 0.3
```

label_md.py 按文件夹赋予 wolf/cat/lion/crocodile 类别，MDV6 只提供框，再按类别重分 8:1:1。未检出动物的图片会跳过，所以最终伪标注图片数可能减少。

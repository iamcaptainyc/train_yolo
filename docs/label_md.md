# 用 MDV6 生成多动物类别伪标注

在仓库根目录运行，权重路径替换为实际的 Ultralytics 兼容 MDV6 .pt 文件。

```bash
python label_md.py \
  --input /root/autodl-tmp/capt/code/train_yolo/datasets/farm_harmful_animals \
  --weights /绝对路径/MDV6.pt \
  --output /root/autodl-tmp/capt/code/train_yolo/yolo_datasets/farm_md_yolo \
  --device 0 --batch 16 --imgsz 1280 --conf 0.3
```

输入可以是 train/类别/图片、validation/类别/图片、test/类别/图片，也可以直接是类别/图片。Label 和 labels 文件夹跳过。

MDV6 提供动物框，动物类别由图片的类别文件夹决定。例如 Wild Boar 文件夹中的全部动物框均写成 wild_boar，Tiger 文件夹中的全部动物框均写成 tiger。类别名转为小写、空格和标点转下划线，按字母序分配从 0 开始的连续 ID，所有实际有检出的文件夹共同组成多类别数据集。混合物种图片需要人工复核。

合并全部原始划分，再按类别重新随机分配 train/val/test 为约 8:1:1，小类别存在整数取整误差。同一 inode 图片去重，未检查内容重复、重编码副本或相邻帧，因此不保证这些图片之间无泄漏。

只保留模型的 animal 类检测。无动物检出的图片跳过，issues.jsonl 记录原因。模型推理失败时停止并保留 INCOMPLETE.txt，使用新输出目录重试。当前没有断点恢复。图片默认硬链接、平铺保存，跨文件系统可以加 --copy-images。

输出包含 images/train、images/val、images/test 和对应 labels，以及 data.yaml、REPORT.md、report.json、selection.jsonl。报告记录输入、权重、输出的绝对路径，逐类及分集图片数和框数。

模型类别名应含 animal 或 animals。类别名被数字替代的权重，确认动物类别的实际 ID 后用 --animal-class-id 指定。预测框使用原图归一化 xywh 坐标，自动限制在图像边界内。

这些是模型伪标注，漏检及误检会影响质量；不把未检出的动物当作已验证负样本。按此划分得到的验证和测试集也使用伪标注，评估真实效果需要人工审核的样本。

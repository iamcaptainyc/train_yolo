# 下载 COCO 2017 动物检测子集

在仓库根目录运行：

```bash
python download_coco.py --classes cow elephant \
  --output yolo_datasets/coco_cow_elephant --workers 16 --retries 5
```

可选类别为 `bird cat dog horse sheep cow elephant bear zebra giraffe`，`cattle` 也可作为 `cow` 的输入别名。脚本从 COCO 下载官方训练和验证标注 ZIP（约 241 MiB），再根据实例框逐张下载匹配的 JPEG。如果已有 ZIP，用 `--annotations-zip /path/to/annotations_trainval2017.zip` 复用。

重跑同一命令会复用完整文件，并对支持 HTTP Range 的服务器续传 `.part`。每张图片只下载一次，保留图中所选类的有效单体框。跳过 `iscrowd` 群体框，超出图像的有效框裁到图像边界。图片保留原 `train2017` 和 `val2017` 划分。

输出含 `images/{train,val}`、`labels/{train,val}`、`data.yaml` 和各类图片数及框数 `report.json`。筛选后的官方 COCO 记录保存在 `_metadata/instances_train2017_subset.json` 和 `_metadata/instances_val2017_subset.json`，保留原始类别 ID 和坐标。下载齐全后才写出 `data.yaml`。改选类别时请指定新输出目录。

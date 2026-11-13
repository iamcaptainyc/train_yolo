# 下载 COCO 2017 动物检测子集

在仓库根目录运行：

```bash
python download_coco.py --classes cow elephant \
  --output yolo_datasets/coco_cow_elephant --workers 16 --retries 5
```

可选类别为 `bird cat dog horse sheep cow elephant bear zebra giraffe`，`cattle` 也可作为 `cow` 的输入别名。脚本默认从 `hf-mirror.com` 下载 COCO 标注 ZIP（约 241 MiB），并通过 HTTP Range 从镜像的 `train2017.zip` 和 `val2017.zip` 中读取 ZIP 目录及命中的图片。已有标注 ZIP 可用 `--annotations-zip /path/to/annotations_trainval2017.zip` 复用。

脚本校验标注 ZIP 的 SHA-256 和从图片 ZIP 中读取的成员 CRC。镜像需要支持 HTTP Range。网络中断时重跑会复用已完成的图片，未完成的图片重新读取 ZIP 对应区间。`--source official` 可使用原来的 COCO 主站和 Google Cloud 下载流程。图片下载失败时，`_metadata/failures.json` 会记录文件名和错误。

重跑同一命令会复用完整文件，并对支持 HTTP Range 的服务器续传标注 ZIP 的 `.part`。开始时按划分显示已完成、待处理和待下载数量。进度条只统计待处理项，并单独显示失败数。镜像频繁断开连接时可降低 `--workers`，继续使用同一个输出目录。每张图片保留图中所选类的有效单体框。跳过 `iscrowd` 群体框，超出图像的有效框裁到图像边界。图片保留原 `train2017` 和 `val2017` 划分。

输出含 `images/{train,val}`、`labels/{train,val}`、`data.yaml` 和各类图片数及框数 `report.json`。筛选后的官方 COCO 记录保存在 `_metadata/instances_train2017_subset.json` 和 `_metadata/instances_val2017_subset.json`，保留原始类别 ID 和坐标。下载齐全后才写出 `data.yaml`。改选类别时请指定新输出目录。

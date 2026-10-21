# SnakeAid 与 Open Images 蛇类数据

核查于 2026-10-21。仅查看仓库元数据、配置及官方文档，没有全量下载两套数据。

## Hugging Face 直接落盘

[SnakeAid 仓库](https://huggingface.co/datasets/the-khiem7/snakeaid-yolov12-5000-bbox)的数据卡列出 5,291 张图片、22 类，使用 `train/valid/test` 的 YOLO 图片和标签目录。网页自动数据预览主要显示标签文本；应通过 `snapshot_download` 下载原始仓库文件。数据卡未独立审计重复图和标注质量。

```bash
git pull --ff-only
python -m pip install -r requirements-huggingface.txt

# 先查看尚需下载的文件数及字节量，估计磁盘需求
python download_huggingface.py the-khiem7/snakeaid-yolov12-5000-bbox --output datasets/snakeaid --dry-run

# 并发下载到指定目录，并生成本地训练配置
python download_huggingface.py the-khiem7/snakeaid-yolov12-5000-bbox --output datasets/snakeaid --workers 8 --retries 5 --snake-yolo-config

python train.py --data datasets/snakeaid/data.local.yaml --model weights/yolo11s.pt --device 0
```

也可只使用官方 SDK：

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="the-khiem7/snakeaid-yolov12-5000-bbox",
    repo_type="dataset",
    local_dir="datasets/snakeaid",
    max_workers=8,
)
print(path)
```

`local_dir` 保留原目录结构，完成状态保存在目录内 `.cache/huggingface/`，重跑复用已有文件。保留输出目录和缓存元数据以便恢复。脚本对临时网络/服务端错误退避重试，存储或权限错误则退出；未完成文件的恢复由官方 SDK 管理，不保证所有传输后端都跨运行逐字节续传。可传 `--revision` 固定完整提交 SHA 以便复现。

仓库实际 `data.yaml` 使用 `../train/images`、`../valid/images`、`../test/images`；直接放在下载根目录时这些路径不适配。`--snake-yolo-config` 会保留原文件，额外写 `data.local.yaml`，设置绝对 `path` 和正确的相对 split 路径，保留 22 类定义。此选项适用于该 SnakeAid 导出布局，通用 HF 下载无需加它。名称中的 YOLOv12 不限制使用 YOLO11/YOLO26；这里是普通 YOLO 检测标注。

官方接口说明：[下载仓库和 local_dir](https://huggingface.co/docs/huggingface_hub/guides/download)。

## Open Images 能否只下载有框的蛇

**可以。** [V7 官方下载页](https://storage.googleapis.com/openimages/web/download_v7.html)区分 bbox 标注与图片级标签，并支持根据图片 ID 清单下载子集。[600 类 bbox 类别表](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv)中存在 `/m/078jl,Snake`。

严格筛选方法：

1. 读取 train/validation/test 的 **Boxes** CSV，而不是图片级标签 CSV。
2. 只保留 `LabelName == '/m/078jl'` 且坐标有效的 bbox 行。
3. 按 ImageID 去重得到图片清单；每行写为 `train/<ImageID>` 等形式。
4. 用官方 `downloader.py` 按清单下载图片，另外保存筛选后的 bbox 标注。

这样只下载有蛇框的图片，不必下载整个 Open Images。train 的全类别 bbox CSV 仍可能很大；图片子集下载不代表元数据也只下载蛇类行。原始 bbox 是归一化 `[XMin,XMax,YMin,YMax]`，不是 YOLO 中心点格式。真实个体检测还应明确处理 `IsGroupOf` 群体框和 `IsDepiction` 绘画/模型等标志，避免直接当普通实体蛇框训练。

也可用官方页面推荐的 FiftyOne 按检测类别下载：

```bash
python -m pip install fiftyone
```

```python
import fiftyone.zoo as foz

if __name__ == "__main__":
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",  # 另运行 validation、test，保持官方 split
        label_types=["detections"],
        classes=["Snake"],
        only_matching=True,
        dataset_dir="datasets/openimages_snake",
        num_workers=8,
    )
    print("带蛇检测框的图片数：", len(dataset))
```

这段代码保留 Open Images/FiftyOne 标注布局，不直接生成 YOLO 数据。FiftyOne 需要其服务运行依赖，详见 [FiftyOne 参数文档](https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html)。默认下载全部符合条件的样本，没有设置随机填充样本上限；不会因为图片有蛇分类标签但没有蛇检测框而选中它。

Open Images 的 `Snake` 是通用蛇类，与 SnakeAid 的 22 个细分类别粒度不同。合并训练前应决定全部统一为单类 snake，或保留细分类标签；不能简单拼接两个来源的 class ID。

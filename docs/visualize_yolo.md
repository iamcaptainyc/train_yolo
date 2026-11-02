# YOLO 检测标注可视化

`visualize_yolo.py` 将真实标签画到图片上，输出 JPEG 预览和 `report.json`，无需模型或 GPU。使用项目已有的 Pillow、PyYAML、tqdm 依赖。

```bash
# 数据集根目录：默认以种子 42 随机抽取 100 张
python visualize_yolo.py --source datasets/animals_yolo

# 按 YAML 读取验证集，抽取 50 张
python visualize_yolo.py --source datasets/animals_yolo/data.yaml --split val --limit 50 --output runs/val_preview

# 导出全部图片，使用数字类别 ID（没有 YAML 也可用）
python visualize_yolo.py --source datasets/animals_yolo/images/train --limit 0 --output runs/train_preview

# 单张图片，额外提供类别名称表
python visualize_yolo.py --source datasets/animals_yolo/images/train/example.jpg --data datasets/animals_yolo/data.yaml --output runs/single_preview

# 自定义图片、标签目录；标签目录保持相对于图片目录的子目录结构
python visualize_yolo.py --source /data/photos --labels /data/annotations --output runs/custom_preview
```

运行后打开输出目录中的 `.jpg` 即可查看。`--output` 默认是 `runs/labels_preview`，必须为新目录，可以放在数据集或图片/标签目录内部；再次运行时使用新的输出路径。输入图片在创建输出目录之前收集，因此本次生成的预览不会加入本次输入；以后递归扫描包含旧预览的图片目录时，旧预览也会被收集。脚本不会修改原图片和标签。输出文件带序号和原文件名，避免不同子目录同名图片互相覆盖；报告保留原图片、标签路径及对应输出名。

支持 `images/{train,val,test}`、`{train,val,test}/images`、平铺 `images/` 和任意图片目录，递归收集 JPG/JPEG/PNG/BMP/WebP/TIF/TIFF。数据集根目录自动寻找 `data.yaml`、`data.yml`、`dataset.yaml`、`dataset.yml`；YAML 中 `names` 可以是列表或 ID→名称字典。直接传图片目录或图片时，可用 `--data` 指定名称表，否则显示数字 ID。

默认将路径中最后一级 `images` 替换为 `labels`，再将扩展名换为 `.txt`；路径中没有 `images` 时查找与图片同目录、同名的 TXT。使用 `--labels` 时，以所选图片目录为根匹配相对子路径；单张图片以其父目录为根。对数据集根目录使用此参数时，也会保留相对于该根目录的完整子路径，通常应直接选择 `images/train` 等图片目录。

`--split all` 收集本地图片树及 YAML 中有效的划分目录/图片清单，失效的 YAML 路径会给出警告；显式 `--split train/val/test` 优先使用 YAML 对应路径，没有配置时从本地目录结构识别。TXT 图片清单中的相对路径以清单目录为基准，同一图片路径会去重。`--limit 0` 导出全部，正整数限制随机抽样数量；`--seed` 控制可复现抽样。

标签仅支持检测格式 `class_id x_center y_center width height`，坐标归一化为 0～1。脚本检查非有限数、越界/退化框和类别表之外的 ID；分割、姿态、旋转框等格式会报错。有问题的整张图跳过，其他图片继续处理。空 TXT 是合法负样本，预览显示 `EMPTY LABEL`；缺失 TXT 显示红色 `MISSING LABEL`，不会当作已确认负样本。缺失标签或处理失败时退出码为 1，并在报告中记录具体路径和原因。

预览最长边默认 1600 像素，不放大小图，可通过 `--max-size` 修改；框宽由 `--line-width` 控制。坐标按原始存储像素绘制，不单独应用 EXIF 旋转，以免框和图片错位。类别颜色固定，标签显示 `ID: 名称`。Windows 优先使用微软雅黑；Linux 中文类别可通过 `--font /path/to/chinese-font.ttf` 指定支持中文的 TTF/TTC 字体。

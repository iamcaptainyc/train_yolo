# YOLO 检测标注可视化

`visualize_yolo.py` 将真实标签画到图片上，输出 JPEG 预览和 `report.json`，无需模型或 GPU。使用项目已有的 Pillow、PyYAML、tqdm 依赖。

```bash
python visualize_yolo.py datasets/animals_yolo
python visualize_yolo.py datasets/animals_yolo --classes fox rabbit
```

直接传数据集目录时，脚本按原划分输出到 `preview/类别/train`、`preview/类别/val`、`preview/类别/test`。默认导出所有有标注框的类别。`--classes` 可传类别名或 ID，控制生成哪些类别目录。多类别图片在各匹配目录各保存一份，每份保留全部框和图例。图像区域最长边为 960 像素，JPEG 质量为 78。无框图片跳过，缺失或损坏的标签及图片错误记入 `preview/report.json`。重新生成前请移走已有的 `preview` 目录。

使用 `--source` 可访问自定义输出、抽样和单图预览模式。

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

# 查看包含左转绿灯标注框的全部图片，画面中只画左转绿灯框
python visualize_yolo.py --source yolo_datasets/sentrymode_data_v2 --classes left_green --output runs/left_green_preview

# 只检查验证集；也可按类别 ID 筛选（ID 应以自己的 data.yaml 为准）
python visualize_yolo.py --source yolo_datasets/sentrymode_data_v2/data.yaml --split val --classes 13 --output runs/class13_val

# 多类别取并集，匹配后抽取 50 张；直接输入图片目录时用 --data 提供名称表
python visualize_yolo.py --source yolo_datasets/sentrymode_data_v2/images/train --data yolo_datasets/sentrymode_data_v2/data.yaml --classes left_green right_green --limit 50 --output runs/arrow_green_sample
```

运行后打开输出目录中的 `.jpg` 即可查看。`--output` 默认是 `runs/labels_preview`，必须为新目录，可以放在数据集或图片/标签目录内部；再次运行时使用新的输出路径。输入图片在创建输出目录之前收集，因此本次生成的预览不会加入本次输入；以后递归扫描包含旧预览的图片目录时，旧预览也会被收集。脚本不会修改原图片和标签。输出文件带序号和原文件名，避免不同子目录同名图片互相覆盖；报告保留原图片、标签路径及对应输出名。

支持 `images/{train,val,test}`、`{train,val,test}/images`、平铺 `images/` 和任意图片目录，递归收集 JPG/JPEG/PNG/BMP/WebP/TIF/TIFF。数据集根目录自动寻找 `data.yaml`、`data.yml`、`dataset.yaml`、`dataset.yml`；YAML 中 `names` 可以是列表或 ID→名称字典。直接传图片目录或图片时，可用 `--data` 指定名称表，否则显示数字 ID。

默认将路径中最后一级 `images` 替换为 `labels`，再将扩展名换为 `.txt`；路径中没有 `images` 时查找与图片同目录、同名的 TXT。使用 `--labels` 时，以所选图片目录为根匹配相对子路径；单张图片以其父目录为根。对数据集根目录使用此参数时，也会保留相对于该根目录的完整子路径，通常应直接选择 `images/train` 等图片目录。

`--split all` 收集本地图片树及 YAML 中有效的划分目录/图片清单，失效的 YAML 路径会给出警告；显式 `--split train/val/test` 优先使用 YAML 对应路径，没有配置时从本地目录结构识别。TXT 图片清单中的相对路径以清单目录为基准，同一图片路径会去重。`--limit 0` 导出全部，正整数限制随机抽样数量；`--seed` 控制可复现抽样。

`--classes` 支持一个或多个非负整数 ID、完整类别名称，也可混用；名称精确匹配，来自数据集 YAML 或额外的 `--data`。同名的多个 ID 都会选中，没有名称表时只能使用数字 ID。只导出至少包含一个所选类别框的图片，同一图片只输出一次，保留完整画面，框和顶部图例仅显示所选类别。指定 `--classes` 后默认导出全部匹配图片；不指定仍默认随机抽取 100 张。显式 `--limit` 在类别筛选之后生效。

类别筛选会先扫描所选 split 的全部标签，空标签和不含所选类别的图片跳过；缺失或非法标签无法可靠判断类别，会记录到 `report.json` 的 `scan_issues` 并以退出码 1 提示，其他匹配图片继续导出。没有匹配图片时只生成零匹配报告。报告的 `total_images` 是筛选前图片数，`matched_images` 是抽样前匹配图片数，`selected_images` 是本次选择导出的图片数；`classes` 记录所选 ID，`counts` 同时包含扫描异常和导出状态。

标签仅支持检测格式 `class_id x_center y_center width height`，坐标归一化为 0～1。脚本检查非有限数、越界/退化框和类别表之外的 ID；分割、姿态、旋转框等格式会报错。有问题的整张图跳过，其他图片继续处理。空 TXT 是合法负样本，预览显示 `EMPTY LABEL`；缺失 TXT 显示红色 `MISSING LABEL`，不会当作已确认负样本。缺失标签或处理失败时退出码为 1，并在报告中记录具体路径和原因。

图片区域最长边默认 1600 像素，不放大小图，可通过 `--max-size` 修改；框宽由 `--line-width` 控制。坐标按原始存储像素绘制，不单独应用 EXIF 旋转，以免框和图片错位。画面内仅绘制固定类别颜色的半透明边框（约 50% 不透明度），框内无填充，不叠加文字。图片上方额外增加独立图例区域，按 ID 排序，仅列出当前图片出现的类别，每类显示一次颜色与 `ID: 名称`；类别较多或名称较长时自动换行。空标签、缺失标签提示也位于顶部区域。最终输出高度包含额外图例，小于 160 像素宽的图片会扩展画布到 160 像素以容纳文字，图片本身不放大。Windows 优先使用微软雅黑；Linux 中文类别可通过 `--font /path/to/chinese-font.ttf` 指定支持中文的 TTF/TTC 字体。

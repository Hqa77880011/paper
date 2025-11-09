🧠 DeepVision Segmentation & Detection Project

本项目包含基于 YOLO、SAM、Transformer 等深度学习模型的目标检测与图像分割实现。
我们提供了完整的数据与训练好的模型，方便你快速测试或二次训练。

📂 项目结构
├── data.zip                # 数据集压缩包（需解压）
├── yolo/                   # YOLO 模型及权重文件
├── sam/                    # SAM (Segment Anything Model) 模型及脚本
├── transformer/            # Transformer 结构模型及相关代码
├── evaluate_segmentation.py # 分割指标计算脚本 (IoU, Dice, Precision, Recall)
└── README.md               # 项目说明文件

📦 数据说明

所有数据存放在 data.zip 文件中。

请先解压该文件：

unzip data.zip -d ./data


解压后，你将获得：

./data/
├── images/       # 原始图像
├── masks/        # 分割标签
└── annotations/  # 检测标签（YOLO 格式）

🧩 模型说明

我们在以下文件夹中提供了训练好的模型权重：

模型类型	文件夹路径	说明
YOLO	yolo/	目标检测模型，支持多类别检测
SAM	sam/	分割模型，可对任意物体生成mask
Transformer	transformer/	基于Transformer的视觉分割模型

你可以直接使用这些模型在你的数据上进行测试，也可以基于我们提供的数据进行重新训练。

🚀 快速开始
1️⃣ 使用我们的模型进行测试
# 示例：使用YOLO模型检测图像
python yolo/test.py --weights yolo/best.pt --source data/images

# 示例：使用SAM模型分割图像
python sam/predict.py --model sam/best.pth --input data/images --output results/

2️⃣ 使用我们的数据训练你自己的模型
# 示例：训练YOLO模型
python yolo/train.py --data data/config.yaml --weights '' --epochs 100


或：

# 示例：训练SAM模型
python sam/train.py --dataset data/ --epochs 50

📈 模型评估

我们提供了一个评估脚本 evaluate_segmentation.py，可计算以下指标：

IoU (Intersection over Union)

Dice 系数

Precision

Recall

使用方式：

python evaluate_segmentation.py \
  --gt-dir "data/masks" \
  --pred-dir "results/masks_pred" \
  --output-dir "results/metrics" \
  --suffix "_mask"

💡 特点

✅ 支持 YOLO / SAM / Transformer 三种模型架构
✅ 提供 预训练模型权重
✅ 提供 可复现的数据集
✅ 适合 自定义数据的训练与验证

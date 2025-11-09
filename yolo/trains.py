import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

data_yaml_path = "/home/Hqa/yolo/yolo11/data.yaml"
pre_model_name = "/home/Hqa/yolo/yolo11/yolo11n.pt"

if __name__ == '__main__':
    model = YOLO(pre_model_name)
    model.train(
        data=data_yaml_path,
        imgsz=360,
        epochs=50,
        batch=64,
        workers=16,
        device='0,1,2,3',
        optimizer='SGD',
        amp=True,

        # ✅ 数据增强策略
        # 几何增强
        degrees=10,        # 随机旋转角度
        translate=0.1,     # 平移比例
        scale=0.3,         # 随机缩放（0.5~1.5）
        shear=5,           # 随机剪切
        flipud=0.3,        # 上下翻转概率
        fliplr=0.5,        # 左右翻转概率

        # 颜色增强
        hsv_h=0.02,       # 色调变化
        hsv_s=0.7,         # 饱和度变化
        hsv_v=0.4,         # 亮度变化

        # 混合增强
        mosaic=1.0,        # 开启 Mosaic（强烈推荐）
        mixup=0.2,         # MixUp 概率
        copy_paste=0.2,    # Copy-Paste 概率

        # 正则化
        label_smoothing=0.05,  # 标签平滑，避免过拟合
    )

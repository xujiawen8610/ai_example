import torch
import torch.nn as nn
import numpy as np
from PIL import Image

"""
model.py - 极简手写数字识别卷积神经网络 (CNN) 与图像预处理模块

【第一性原理认知】：
1. 图像本质：28x28 的灰度图像本质上是一个 28 行、28 列的二维数字矩阵，每个数值在 0~255 之间代表灰度亮度。
2. 卷积层 (Conv2d)：局部特征提取器。通过在图像上滑动 3x3 小窗口（卷积核），感知并提取笔画的边缘、拐角和端点等局部几何模式。
3. 池化层 (MaxPool2d)：空间降采样。取局部区域的最大值，既缩小了特征图尺寸、压缩计算量，又增强了模型对数字微小位移的平移不变性。
4. 全连接层 (Linear)：最终决策层。将提取到的高维特征展平并综合，映射为 0~9 共 10 个数字分类的未归一化得分 (Logits)。
"""

class DigitCNN(nn.Module):
    """
    轻量双层卷积神经网络
    结构：Conv2d(1->16) -> ReLU -> MaxPool -> Conv2d(16->32) -> ReLU -> MaxPool -> Linear(1568->10)
    总参数量仅约 2 万个，在 CPU 上可在数秒内完成训练。
    """
    def __init__(self):
        super().__init__()
        # 特征提取骨架
        self.features = nn.Sequential(
            # 第一层卷积：输入 1 通道灰度图，提取 16 种初级笔画特征，输出尺寸保持 28x28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 首次池化：尺寸减半为 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积：输入 16 通道，组合提取 32 种复合模式特征，输出尺寸保持 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 二次池化：尺寸减半为 7x7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类输出层：32 * 7 * 7 = 1568 个神经元特征输入，输出 10 个数字的得分
        self.classifier = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        :param x: 形状为 (BatchSize, 1, 28, 28) 的图像张量
        :return: 形状为 (BatchSize, 10) 的数字得分张量 (Logits)
        """
        out = self.features(x)
        out = out.flatten(start_dim=1)  # 展平为 (BatchSize, 1568)
        logits = self.classifier(out)   # 映射为 10 分类 logits
        return logits


def preprocess_image(image_input) -> torch.Tensor:
    """
    图像预处理函数 (符合 MNIST 官方制作标准):
    1. 使用 Pillow 打开任意尺寸、格式的图片（支持文件路径或 PIL Image 对象）。
    2. 处理透明通道：若包含透明背景，根据不透明笔画深浅自动合成对应底色。
    3. 自适应底色校正（第一性原理）：
       - 采样图像四周边缘估算背景基准灰度 Bg。
       - 若属于浅色底暗字（白纸黑字），提取反差信号 (Bg - 像素值)；
       - 若属于黑底亮字，提取高亮信号 (像素值 - Bg)。
    4. 紧贴外接矩形裁剪 (Bounding Box) + 居中对齐：
       - 去除任意长宽比画布四周多余的空白留白，提取紧凑的纯数字笔迹。
       - 保持宽高比等比缩放至 20x20 框内，防止字形被拉扁或挤瘦。
       - 居中粘贴至 28x28 画布中心（四周留 4 像素安全边距），完美对齐 MNIST 训练集特征分布。
    5. 归一化至 [0.0, 1.0] 区间，输出 4D 张量 (1, 1, 28, 28)。
    """
    # 1. 读取图像并处理透明通道 (RGBA / LA / 调色板透明)
    if isinstance(image_input, Image.Image):
        img = image_input.copy()
    else:
        img = Image.open(image_input)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.array(rgba.split()[-1])
        if alpha.min() < 255:
            gray = np.array(rgba.convert("L"))
            opaque_mask = alpha > 128
            ink_is_dark = (gray[opaque_mask].mean() < 128.0) if opaque_mask.any() else True
            bg_color = (255, 255, 255, 255) if ink_is_dark else (0, 0, 0, 255)
            bg = Image.new("RGBA", rgba.size, bg_color)
            img = Image.alpha_composite(bg, rgba).convert("L")
        else:
            img = rgba.convert("L")
    else:
        img = img.convert("L")

    # 2. 转为 NumPy 数组并估算背景基准灰度 Bg
    arr = np.array(img, dtype=np.float32)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = border.mean()

    # 3. 提取有效笔画反差信号 (转换为黑底白字高亮信号)
    if bg > 127.0 or bg > arr.mean():
        sig = np.maximum(0.0, bg - arr)
    else:
        sig = np.maximum(0.0, arr - bg)

    # 4. 标准化为 28x28 特征矩阵：
    # - 若原生已经是 28x28 标准图（如测试集样本），直接归一化并保留原生笔触结构；
    # - 若为外部任意尺寸大图/手写涂鸦，则提取外接矩形紧贴裁剪，等比缩放至 20x20 并居中对齐（符合 MNIST 官方制作标准）。
    if img.size == (28, 28):
        max_val = sig.max()
        final_arr = (sig / max_val) if max_val > 0.0 else sig
    else:
        threshold = sig.max() * 0.15 if sig.max() > 0 else 0
        coords = np.argwhere(sig > threshold)

        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # 裁剪出紧贴数字的矩形区域
            cropped_sig = sig[y_min:y_max + 1, x_min:x_max + 1]
            cropped_norm = (cropped_sig / cropped_sig.max() * 255.0).astype(np.uint8)
            cropped_img = Image.fromarray(cropped_norm)

            # 保持纵横比等比缩放至最大 20x20
            cw, ch = cropped_img.size
            if cw > ch:
                new_w = 20
                new_h = max(1, int(round(ch * 20.0 / cw)))
            else:
                new_h = 20
                new_w = max(1, int(round(cw * 20.0 / ch)))

            resized_digit = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # 粘贴在 28x28 纯黑画布中心 (居中留白，对齐 MNIST 规范)
            canvas = Image.new("L", (28, 28), 0)
            pos_x = (28 - new_w) // 2
            pos_y = (28 - new_h) // 2
            canvas.paste(resized_digit, (pos_x, pos_y))

            final_arr = np.array(canvas, dtype=np.float32) / 255.0
        else:
            final_arr = np.zeros((28, 28), dtype=np.float32)

    # 5. 构造 PyTorch 张量 (1, 1, 28, 28)
    tensor = torch.from_numpy(final_arr).unsqueeze(0).unsqueeze(0)
    return tensor


import os
import sys
import argparse
import torch
from model import LinearToyModel

"""
predict.py - 单样本独立测试 / 推理脚本

【核心任务】：
解决新手疑问：“如果我只想单独测试某一个特定输入（如 x1=10, x2=2），该怎么做？”
本脚本展示如何将一组特定的数值送入训练好的模型中进行推理。

【运行命令】：
1. 默认单样本测试 (x1=5.0, x2=1.0):
   python predict.py

2. 命令行直接指定数值:
   python predict.py 10 2

3. 命令行命名参数测试:
   python predict.py --x1 10 --x2 2

4. 终端交互模式 (连续手动输入数字进行测试):
   python predict.py -i
"""

# 获取当前脚本所在目录，确保在任何工作路径下都能正确找到 best_model.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

def load_model(model_path=MODEL_PATH):
    """加载已经训练好的模型权重"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型权重文件 '{model_path}'！\n请先运行训练脚本生成权重: python train.py"
        )

    # 1. 创建模型结构骨架
    model = LinearToyModel()

    # 2. 读取磁盘上的权重数据
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path)

    # 3. 将权重装载进模型
    model.load_state_dict(state_dict)
    model.eval()  # 设为评估模式
    return model

def predict_one(model, x1: float, x2: float):
    """对单组 (x1, x2) 进行推理预测并打印结果"""
    # 将输入的两个标量包装成形状为 (1, 2) 的二维张量 (batch_size=1, feature_dim=2)
    x_tensor = torch.tensor([[x1, x2]], dtype=torch.float32)

    # 推理计算（不需要计算梯度）
    with torch.no_grad():
        output = model(x_tensor)
        prediction = output.item()

    # 理论公式真实值计算：y = 2*x1 + 3*x2 + 1
    ground_truth = 2.0 * x1 + 3.0 * x2 + 1.0
    error = abs(prediction - ground_truth)

    print("-" * 55)
    print(f"输入特征:          x1 = {x1}, x2 = {x2}")
    print(f"理论公式真实输出:  2 * {x1} + 3 * {x2} + 1 = {ground_truth:.4f}")
    print(f"模型预测输出:      {prediction:.4f}")
    print(f"误差 (绝对差值):   {error:.6f}")
    print("-" * 55)
    return prediction

def interactive_mode(model):
    """交互式测试模式"""
    print("=" * 55)
    print("  进入交互式测试模式 (输入 'q' 退出)")
    print("=" * 55)
    while True:
        try:
            input_x1 = input("\n请输入 x1: ").strip()
            if input_x1.lower() == 'q':
                print("已退出交互测试。")
                break
            input_x2 = input("请输入 x2: ").strip()
            if input_x2.lower() == 'q':
                print("已退出交互测试。")
                break

            val_x1 = float(input_x1)
            val_x2 = float(input_x2)
            predict_one(model, val_x1, val_x2)
        except ValueError:
            print("输入无效！请输入合法的数字或小数。")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出交互测试。")
            break

def main():
    parser = argparse.ArgumentParser(
        description="单样本单独测试脚本",
        epilog="使用示例:\n"
               "  python predict.py            # 默认测试 x1=5.0, x2=1.0\n"
               "  python predict.py 10 2       # 直接传入两数测试\n"
               "  python predict.py --x1 10 --x2 2  # 命名参数测试\n"
               "  python predict.py -i         # 终端交互模式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("coords", nargs="*", type=float, default=[], help="输入特征 x1 x2 (可选，例如: python predict.py 10 2)")
    parser.add_argument("--x1", type=float, default=None, help="输入特征 x1 的数值")
    parser.add_argument("--x2", type=float, default=None, help="输入特征 x2 的数值")
    parser.add_argument("--interactive", "-i", action="store_true", help="开启终端交互输入测试模式")

    args = parser.parse_args()

    try:
        model = load_model()
    except FileNotFoundError as err:
        print(f"【错误】: {err}")
        return

    if args.interactive:
        interactive_mode(model)
        return

    # 判断传参方式
    if len(args.coords) == 2:
        val_x1, val_x2 = args.coords[0], args.coords[1]
    elif args.x1 is not None and args.x2 is not None:
        val_x1, val_x2 = args.x1, args.x2
    elif len(args.coords) == 0 and args.x1 is None and args.x2 is None:
        # 默认使用标准标杆测试值
        val_x1, val_x2 = 5.0, 1.0
    else:
        print("【提示】 单独测试需要提供 2 个数值 (x1 和 x2)，例如：")
        print("  python predict.py 10 2")
        print("  python predict.py --x1 10 --x2 2")
        print("或者直接运行以使用默认值 (5.0, 1.0)：")
        print("  python predict.py")
        return

    print("=" * 55)
    print(f"  执行单样本测试 (x1={val_x1}, x2={val_x2})")
    print("=" * 55)
    predict_one(model, val_x1, val_x2)
    print("\n【提示】你也可以通过命令行传入任意数值单独测试，例如：")
    print("  python predict.py 10 2")
    print("或者进入交互模式：")
    print("  python predict.py -i")

if __name__ == "__main__":
    main()

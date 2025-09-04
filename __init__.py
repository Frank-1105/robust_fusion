import torch
import torch.nn as nn

from generator import GeneratorResnet



class GeneratorResnetWrapper(nn.Module):
    """
    一个封装类，用于处理两个3通道图像输入，并将其传递给GeneratorResnet。
    """
    def __init__(self, inception=False, eps=1.0, evaluate=False):
        super(GeneratorResnetWrapper, self).__init__()
        # 在初始化时，创建原始的GeneratorResnet实例
        self.generator = GeneratorResnet(inception=inception, eps=eps, evaluate=evaluate)

    def forward(self, x, y, grad):
        """
        拼接两个3通道张量（x和y），然后将其传递给生成器。
        
        参数：
            x (torch.Tensor): 第一个输入张量，形状为 (B, 3, H, W)。
            y (torch.Tensor): 第二个输入张量，形状为 (B, 3, H, W)。
            grad (torch.Tensor): 用于生成器内部逻辑的标量张量。
            
        返回：
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            从生成器模型返回的四个张量。
        """
        # 检查输入张量是否为3通道
        if x.shape[1] != 3 or y.shape[1] != 3:
            raise ValueError("两个输入张量都必须有3个通道。")

        # 沿着通道维度（dim=1）拼接x和y
        input_tensor = torch.cat((x, y), dim=1)

        # 将拼接后的张量和grad传递给原始生成器模型
        return self.generator(input_tensor, grad)


# --- 使用示例 ---
if __name__ == "__main__":
    # 确保上面定义了原始的GeneratorResnet和ResidualBlock类
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    # 实例化封装类
    model = GeneratorResnetWrapper(inception=False).to(device)

    # 创建两个随机的3通道输入张量，模拟您的场景
    x = torch.randn(2, 3, 800, 600).to(device)
    y = torch.randn(2, 3, 800, 600).to(device)
    
    # 创建一个grad张量
    grad_tensor = torch.tensor(0.5)

    # 将两个张量传递给封装类的forward方法
    with torch.no_grad():
        x_inf, x_0, x_out, grad_img ,mask = model(x, y, grad_tensor)

    # 打印输出张量的形状以进行验证
    print("\n输出张量形状:")
    print(f"x_inf 形状: {x_inf.shape}")
    print(f"x_0 形状: {x_0.shape}")
    print(f"x_out 形状: {x_out.shape}")
    print(f"mask 形状: {mask.shape}")
    print(f"grad_img 形状: {grad_img.shape}")
import torch
import torch.nn as nn

ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, eps=1.0, evaluate=False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception

        # Input_size = 6, h, w
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(6, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 6, h, w
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 6, h/2, w/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 6, h/4, w/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 6, h/4, w/4
        self.upsampl_inf1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 6, h/2, w/2
        self.upsampl_inf2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 6, h, w
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        # Input size = 6, h/4, w/4
        self.upsampl_01 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 6, h/2, w/2
        self.upsampl_02 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 6, h, w
        self.blockf_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 1, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

        self.eps = eps
        self.evaluate = evaluate


    def forward(self, input, grad):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        code = x

        ## 分支一，对应强度图
        x = self.upsampl_inf1(code)
        x = self.upsampl_inf2(x)
        x = self.blockf_inf(x)
        if self.inception:
            x = self.crop(x)
        x_inf = self.eps * torch.tanh(x)  # [-eps, eps]

        ## 分支二，对应mask   
        x = self.upsampl_01(code)
        x = self.upsampl_02(x)
        x = self.blockf_0(x)
        if self.inception:
            x = self.crop(x)
        x = (torch.tanh(x) + 1) / 2  # [0, 1]

        # 根据分支二的输出，生成mask
        if self.evaluate:
            # 根据 x 是否小于 0.5 来生成 mask： 真为 0，假为 1
            # .detach() 用于阻止梯度传播
            x_0 = torch.where(x < 0.5, torch.zeros_like(x).detach(), torch.ones_like(x).detach())
        else:
            # 训练模式下，
            # 先生成一个与 x 形状相同的随机张量，其中的每个值都介于 [0, 1] 之间。如果随机值小于 0.5，则条件为真

            # 如果 condition 为 True（随机数小于 0.5），那么 x_0 的对应像素值将直接取自模型上采样分支2的输出 x
            # 如果 condition 为 False（随机数大于等于 0.5），那么 x_0 的对应像素值将由这个嵌套的 torch.where 语句决定
            # 内层的 torch.where 检查 x 是否小于 0.5，如果是，则将对应像素值设为 0，否则设为 grad 的值
            # 当 x_0 由内层决定时，不传播梯度
            x_0 = torch.where(torch.rand(x.shape).cuda()< 0.5, x,
                              torch.where(x < 0.5, torch.zeros_like(x), torch.ones_like(x)*grad).detach())



        # x_out = torch.clamp((x_inf * x_0) + input, min=0, max=1)
        grad_img = torch.clamp(grad * input, min=0, max=1)
        mask = x_inf * x_0
        # return x_out, x_inf, x_0, x, grad_img
        return  x_inf, x_0, x, grad_img, mask
'''
    x_inf: 强度图，形状为 [batch_size, 3, h, w]，值在 [-eps, eps] 范围内
    x_0: mask，形状为 [batch_size, 1, h, w]，值为 0 或 grad --> 是根据分支二的输出生成的mask  用 sparsity loss限制
    x: mask，形状为 [batch_size, 1, h, w] --> 是分支二的原始输出，值在 [0, 1] 范围内 用 quantization loss 限制
    grad_img: 形状为 [batch_size, 6, h, w]，是输入图像乘以 grad 的结果
    mask： mask = x_inf * x_0
'''


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
    
if __name__ == "__main__":
    # 创建模型实例
    model = GeneratorResnet()

    input_tensor = torch.randn(2, 6, 400, 600)
    
    # 检查是否有可用的 CUDA 设备，并把输入张量移动到 GPU 上
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
        print("Using CUDA device.")
    else:
        print("CUDA not available. Using CPU.")

    # grad 是一个用于训练模式的张量，这里我们随便创建一个标量
    grad_tensor = torch.tensor(0.5)


    # 验证模型输出
    with torch.no_grad():
         x_inf, x_0, x, grad_img ,mask = model(input_tensor, grad_tensor)

    # 打印输出张量的形状来验证
    print("Output shapes:")
    print(f"x_inf shape: {x_inf.shape}")  # 期望输出形状: [2, 3, 400, 600]
    print(f"x_0 shape: {x_0.shape}")      # 期望输出形状: [2, 1, 400, 600]
    print(f"x shape: {x.shape}")          # 期望输出形状: [2, 1, 400, 600]
    print(f"mask shape: {mask.shape}")  # 期望输出形状: [2, 3, 400, 600]
    print(f"grad_img shape: {grad_img.shape}") # 期望输出形状: [2, 6, 400, 600]

  


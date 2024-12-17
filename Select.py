import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange


class Select(nn.Module):
    def __init__(self, num=1536):
        super(Select, self).__init__()
        self.linear = nn.Linear(num, num)
        self.linear1 = nn.Linear(num, num)
        self.pool_h = nn.AdaptiveAvgPool1d(1)
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        self.linear_max = nn.Linear(num, num)
        self.linear_Aug = nn.Linear(num, num)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=1536)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=1536)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        # x_M, _ = torch.max(x, dim=-1)
        x = self.linear(x)
        x_M = self.pool_h(x.transpose(1, 2))
        x_M = x_M.squeeze(dim=-1)
        x_M = self.linear_max(x_M)
        x_M = self.gn1(x_M)
        # x_M = self.softmax(x_M)
        # x_M = torch.matmul(x, x_M)
        x_M = x_M.unsqueeze(dim=1)
        x_A = self.pool_w(x.transpose(1, 2))
        x_A = x_A.squeeze(dim=-1)
        x_A = self.linear_Aug(x_A)
        x_A = self.gn2(x_A)
        # x_A = self.softmax(x_A)
        x_A = x_A.unsqueeze(dim=1)

        x = self.linear1(x)

        out1 = x@ rearrange(x_M, 'b l c -> b c l')
        out2 = x@rearrange(x_A, 'b l c -> b c l')
        out1 = out1.softmax(dim=1)
        out2 = out2.softmax(dim=1)
        out = out1*out2
        out = out.squeeze(dim=-1)

        topk_indices = torch.topk(out, k=10, dim=1)[1]

        # 使用索引选择对应的数值张量
        selected_values = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
# class CoordAtt(nn.Module):
#     def __init__(self, num=1024):
#         super(CoordAtt, self).__init__()
#         self.linear = nn.Linear(num, num)
#         self.linear1 = nn.Linear(num, num)
#         self.pool_h = nn.AdaptiveAvgPool1d(1)
#         # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.pool_w = nn.AdaptiveMaxPool1d(1)
#         self.linear_max = nn.Linear(num, num)
#         self.linear_Aug = nn.Linear(num, num)
#         self.gn1 = nn.GroupNorm(num_groups=32, num_channels=1024)
#         self.gn2 = nn.GroupNorm(num_groups=32, num_channels=1024)
#         self.softmax = nn.Softmax(dim=-1)
#
#
#     def forward(self, x):
#
#         # x_M, _ = torch.max(x, dim=-1)
#         x = self.linear(x)
#         x_M = self.pool_h(x.transpose(1, 2))
#         x_M = x_M.squeeze(dim=-1)
#         x_M = self.linear_max(x_M)
#         x_M = self.gn1(x_M)
#         # x_M = self.softmax(x_M)
#         # x_M = torch.matmul(x, x_M)
#         x_M = x_M.unsqueeze(dim=1)
#         x_A = self.pool_w(x.transpose(1, 2))
#         x_A = x_A.squeeze(dim=-1)
#         x_A = self.linear_Aug(x_A)
#         x_A = self.gn2(x_A)
#         # x_A = self.softmax(x_A)
#         x_A = x_A.unsqueeze(dim=1)
#
#         x = self.linear1(x)
#
#         out1 = x@ rearrange(x_M, 'b l c -> b c l')
#         out2 = x@rearrange(x_A, 'b l c -> b c l')
#         out1 = out1.softmax(dim=1)
#         out2 = out2.softmax(dim=1)
#         out = out1*out2
#         out = out.squeeze(dim=-1)
#
#         topk_indices = torch.topk(out, k=20, dim=1)[1]
#
#         # 使用索引选择对应的数值张量
#         selected_values = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
#
#
#
        return selected_values

if __name__ == '__main__':

    x = torch.randn(1, 95, 1536)
    model = Select()

    out = model(x)
    print(out.shape)

# import torch
#
#
# class FrameCrossMultiply(nn.Module):
#     def __init__(self, input_dim):
#         super(FrameCrossMultiply, self).__init__()
#         self.input_dim = input_dim
#         self.linear = nn.Linear(input_dim, input_dim)
#
#     def forward(self, x):
#         # x 的形状为 (b, t, input_dim)，首先对输入进行调整以便进行矩阵乘法
#         x = x.unsqueeze(2)  # 在第二维度上增加一个维度，形状变为 (b, t, 1, input_dim)
#
#         # 对每个样本的每个时间步，与其中的一帧进行矩阵乘法
#         x_cross = torch.matmul(x, self.linear.weight.unsqueeze(0).unsqueeze(0))
#
#         # 返回结果并去除额外的维度
#         return x_cross.squeeze(3)

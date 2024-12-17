from functools import partial
from collections import OrderedDict
from sk_att import *
import torch
import torch.nn as nn
from einops import rearrange, repeat
import os
from Res_model import ResNet, Bottleneck, resnet50
# from Asycross_Atten1 import CrossAttention
os.environ['CUDA_VISIBLE_DEVICES']='0'


class Select(nn.Module):
    def __init__(self, num=1024):
        super(Select, self).__init__()
        self.linear = nn.Linear(num, num)
        self.linear1 = nn.Linear(num, num)
        self.pool_h = nn.AdaptiveAvgPool1d(1)
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        self.linear_max = nn.Linear(num, num)
        self.linear_Aug = nn.Linear(num, num)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=1024)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=1024)
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
        return selected_values


class Dstanet(nn.Module):
    def __init__(self, num_class):
        super(Dstanet, self).__init__()
        config = [[64, 64, 16, 1], [64, 64, 16, 1],
          [64, 128, 32, 1], [128, 128, 32, 1],
          [128, 256, 64, 1], [256, 256, 64, 1],
          [256, 256, 64, 1], [256, 256, 64, 1],
          ]
        self.DSTA_net_body = DSTANet(config=config, num_point=25)# .cuda()
        self.DSTA_net_hand = DSTANet(config=config, num_point=42)# .cuda()
        self.Linear = nn.Linear(in_features=1024, out_features=num_class)
        self.soft = nn.Softmax(dim=1)  # softmax
        self.Asycross = CrossAttention(dim=256, input_1=256, input_2=256, out_dim=256)
        # self.Resnet = torch.nn.DataParallel(resnet50(end2end=False, pretrained=False))
        self.Linear1 = nn.Linear(512, 1024)
        self.Select = Select(num=1024)



    def forward(self, frames, faces, bodies, l_hand, r_hand):
        # batch = frames.shape[0]
        # print(batch)
        # ================images=======================
        # frames = rearrange(frames, 'b t c w h -> (b t) c w h')
        # x = self.Resnet(frames)
        # x = rearrange(x, '(b t ) c ->b t c', b=batch)
        # face_out = x.mean(dim=1)
        body = bodies
        body = body.view(body.size(0), body.size(1), -1, 3)
        body_x = body[:, :, :, 0].clone()
        body_y = body[:, :, :, 1].clone()
        body_confidences = body[:, :, :, 2].clone()
        # todo 关节置信度阈值，设为 0.1, 修改为0.3
        # t = torch.Tensor([self.confidence_threshold]).cuda()  # threshold for confidence of joints TODO .cuda()
        # 通过阈值的值变为1，否则变为0
        t = 0.1
        body_confidences = (body_confidences > t).float() * 1  # TODO .float()去掉
        # make all joints with threshold lower than
        # features_positions :Tensor(b, 95, 70, 3)
        body = torch.stack(
            (body_x * body_confidences, body_y * body_confidences), dim=3)
        body_points = self.DSTA_net_body(body)  # (b*t, 256)
        # body_points = rearrange(body, 'b t c i -> b t (c i)', b=batch)   # TODO
        #         # ===========================================================
        # body_out = body_points.mean(dim=1)

        hand = torch.cat((l_hand, r_hand), dim=2)
        hand = hand.view(hand.size(0), hand.size(1), -1, 3)
        hand_x = hand[:, :, :, 0].clone()
        hand_y = hand[:, :, :, 1].clone()
        hand_confidences = hand[:, :, :, 2].clone()
        # todo 关节置信度阈值，设为 0.1, 修改为0.3
        # t = torch.Tensor([self.confidence_threshold]).cuda()  # threshold for confidence of joints TODO .cuda()
        # 通过阈值的值变为1，否则变为0
        t = 0.1
        hand_confidences = (hand_confidences > t).float() * 1  # TODO .float()去掉
        # make all joints with threshold lower than
        # features_positions :Tensor(b, 95, 70, 3)     torch.Size([1, 95, 42, 2])
        hand = torch.stack(
            (hand_x * hand_confidences, hand_y * hand_confidences), dim=3)
        hand_points = self.DSTA_net_hand(hand)
        hand_body = self.Asycross(hand_points, body_points)
        hand_body = self.Linear1(hand_body)
        select_hand_body = self.Select(hand_body)

        select_hand_body_out = select_hand_body.mean(dim=1)
        # hand_body_out = hand_body.mean(dim=1)
        # out = 0.5*hand_body_out + 0.5*face_out
        # face_hand_body = self.Asycross2(x, hand_body)

        # hand_out = hand_points.mean(dim=1)

        # out = 0.5*body_out + 0.5*hand_out

        out = self.Linear(select_hand_body_out)

        return out


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 input_1=84,
                 input_2=50,
                 out_dim=256,
                 qkv_bias=False,
                 kv_bias=False,
                 q_bias=False,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.,
                 drop=0.5):
        # # super()继承父类的构造函数
        super().__init__()
        self.num_heads = num_heads

        # 对每一个head的dim
        head_dim = dim // num_heads
        # 如果传入了qk_scale,就用传入的，
        # 否则使用每个头维度乘-0.5次方，对应attention公式中（QKT/根号d）*V的分母
        self.scale = qk_scale or head_dim ** -0.5
        # 定义q k v 的初始值
        # 使用一个全连接层直接得到qkv，全连接层节点个数*3=使用3个节点个数为dim的全连接层
        # 此处这样做的目的可能是为了并行化
        self.linear_1 = nn.Linear(input_1, out_dim, bias=q_bias)
        self.linear_2 = nn.Linear(input_2, out_dim, bias=q_bias)
        self.q1 = nn.Linear(head_dim, head_dim*2, bias=kv_bias)
        self.q2 = nn.Linear(head_dim, head_dim*2, bias=kv_bias)
        self.k2 = nn.Linear(head_dim, head_dim * 2, bias=kv_bias)

        self.qkv1 = nn.Linear(dim, dim * 3, bias=kv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=kv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # 再定义一个全连接层，对应多头自注意输出b Concate拼接后，乘的W0 。输入输出节点个数都等于dim
        self.proj1 = nn.Linear(2*dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)
        # self.fc1 = nn.Sequential(nn.Linear(256, 768), nn.ReLU())
        # self.conv = nn.Conv1d(in_channels=256, out_channels=768, kernel_size=1, stride=1)
        self.drop = nn.Dropout(p=drop)
        # torch.Size([1, 95, 84])
        # torch.Size([1, 95, 50])
    def forward(self, x, y):
        x = self.linear_1(x) # torch.Size([1, 95, 256])
        y = self.linear_2(y) # torch.Size([1, 95, 256])
        # x = x.unsqueeze(1) # torch.Size([1, 1, 95, 84])
        # y = y.unsqueeze(2) # torch.Size([1, 95, 1, 50])
        # y = self.conv(y)
        # y = rearrange(y, 'b c 1 -> b 1 c')
        Bx, Nx, Cx = x.shape  # (b, t, 256) ([1, 95, 256])
        By, Ny, Cy = y.shape  # (b, t, 256) ([1, 95, 256])
        #q1 = self.q1(x).reshape(Bx, Nx, 1, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)  # (b, 95, 1, 4, 196)    torch.Size([1, 1, 8, 95, 32])
        qkv1 = self.qkv1(x).reshape(By, Ny, 3, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)  # (b, 95, 2, 4, 196)  torch.Size([2, 1, 8, 95, 32])
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # (b, 4, 1, 196) # make torchscript happy (cannot use tensor as tuple)
        # torch.Size([1, 8, 95, 32])  torch.Size([1, 8, 95, 32])  torch.Size([1, 8, 95, 32])
        qkv2 = self.qkv2(y).reshape(By, Ny, 3, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)  # (b, 95, 2, 4, 196)  torch.Size([2, 1, 8, 95, 32])
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # (b, 4, 1, 196) # make torchscript happy (cannot use tensor as tuple)
        q1 = self.q1(q1)
        q2 = self.q2(q2)
        k_new, v_new = torch.cat([k1, k2], dim=-1), torch.cat([v1, v2], dim=-1) #

        kv_new1 = (q1 @ k_new.transpose(-2, -1)) * self.scale
        kv_new1 = kv_new1.softmax(dim=-1)
        kv_new1 = self.attn_drop(kv_new1)
        kv_new1 = (kv_new1 @ v_new).transpose(1, 2).reshape(By, Nx, Cy*2)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        kv_new1 = self.proj1(kv_new1)
        # 在进行一次dropout得到最终输出
        kv_new1 = self.proj_drop1(kv_new1)  # (b, 1, 768)
        # kv = rearrange(kv, '(b f) p c -> b f p c',  f=323)
        x1 = x + kv_new1

        # kv_new2 = (q2 @ k_new.transpose(-2, -1)) * self.scale
        # kv_new2 = kv_new2.softmax(dim=-1)
        # kv_new2 = self.attn_drop(kv_new2)
        # kv_new2 = (kv_new2 @ v2).transpose(1, 2).reshape(By, Nx, Cy)
        # # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        # kv_new2 = self.proj2(kv_new2)
        # # 在进行一次dropout得到最终输出
        # kv_new2 = self.proj_drop2(kv_new2)  # (b, 1, 768)
        # # kv = rearrange(kv, '(b f) p c -> b f p c',  f=323)
        # y1 = y + kv_new2

        k2 = self.k2(k2)
        kv_new2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        kv_new2 = kv_new2.softmax(dim=-1)
        kv_new2 = self.attn_drop(kv_new2)
        kv_new2 = (kv_new2 @ v2).transpose(1, 2).reshape(By, Nx, Cy)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        kv_new2 = self.proj2(kv_new2)
        # 在进行一次dropout得到最终输出
        kv_new2 = self.proj_drop2(kv_new2)  # (b, 1, 768)
        # kv = rearrange(kv, '(b f) p c -> b f p c',  f=323)
        y1 = y + kv_new2

        # cro = x1+y1
        q1 = rearrange(q1, 'b h t d -> b t (h d)')
        cro = torch.cat([x1, y1], dim=2)
        cro = cro + q1
        return cro
def Dstanet_net(num_classes: int = 11, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    # 调用VisionTransformer（）这个类，根据不同的配置传入不同的参数
    model = Dstanet(num_class=num_classes)
    return model

"""
@Time : 2024/6/12 15:29
@Auth : Thirteen
@File : UniM2CT.py
@IDE : PyCharm
@Function : 
"""

import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class ExtraFreq(nn.Module):
    def __init__(self, img_size, out_c):
        super(ExtraFreq, self).__init__()
        self.img_size = img_size
        self.timg = torch.zeros((img_size, img_size))
        self.low_f = torch.zeros((img_size, img_size))
        self.high_f = torch.ones((img_size, img_size))
        self.mid_f = torch.ones((img_size, img_size))

        self.dct_filter = self.get_DCTf(img_size=img_size)
        self.low_f, self.mid_f, self.high_f = self.create_f(img_size=img_size)

        self.block = nn.Sequential(
            nn.Conv2d(3, int(out_c / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_c / 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(int(out_c / 2), out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def get_DCTf(self, img_size):
        M, N = img_size, img_size
        timg = torch.zeros((M, N))
        timg[0, :] = 1 * torch.sqrt(torch.tensor(1 / N))
        for i in range(1, M):
            for j in range(N):
                timg[i, j] = (torch.cos(torch.tensor(torch.pi * i * (2 * j + 1) / (2 * N)))
                              * torch.sqrt(torch.tensor(2 / N)))
        return timg

    def create_f(self, img_size):
        resolution = np.array((img_size, img_size))
        t_1 = resolution // 16
        t_2 = resolution // 8
        low_f = torch.zeros((img_size, img_size))
        high_f = torch.ones((img_size, img_size))
        mid_f = torch.ones((img_size, img_size))
        for i in range(t_1[0]):
            for j in range(t_1[1] - i):
                low_f[i, j] = 1
        for i in range(t_2[0]):
            for j in range(t_2[1] - i):
                high_f[i, j] = 0
        mid_f = mid_f - low_f
        mid_f = mid_f - high_f
        return low_f, mid_f, high_f

    def DCT(self, img):
        dst = self.dct_filter.to(img.device) * img
        dst = dst * self.dct_filter.to(img.device).permute(0, 1)
        return dst

    def IDCT(self, img):
        dst = self.dct_filter.to(img.device).permute(0, 1) * img
        dst = dst * self.dct_filter.to(img.device)
        return dst

    def forward(self, img):
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        dct_1, dct_2, dct_3 = self.DCT(r), self.DCT(g), self.DCT(b)

        fl = [self.low_f.to(img.device), self.mid_f.to(img.device), self.high_f.to(img.device)]
        re = []
        for i in range(3):
            t_1 = dct_1 * fl[i]
            t_2 = dct_2 * fl[i]
            t_3 = dct_3 * fl[i]
            re.append(self.IDCT(t_1 + t_2 + t_3))
        out = torch.cat((re[0].unsqueeze(1), re[1].unsqueeze(1), re[2].unsqueeze(1)), dim=1)

        out = self.block(out)
        return out

class PatchEmbed(nn.Module):
    """
    img_size : the size of feature map extracted by spatial domain
    patch_size : the size of the patch using on embeding
    in_c : the channel of the feature map extracted by spatial domain
    embed_dim:the embedding dimension
    """

    def __init__(self, img_size=56, patch_size=56, in_c=32, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # [b, p_num, embed_dim]


class ATT(nn.Module):
    """
    in_c : the channel of the feature map extracted by spatial domain
    patch_size : the size of the patch using on embedding
    dim : the embedding dimension
    attn_drop_ratio : the dropout rate of the attention
    proj_drop_ratio : the dropout rate of the projection
    """

    def __init__(self, in_c, patch_size, dim, num_heads=8,
                 qkv_bias=False, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(ATT, self).__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = (patch_size * patch_size * in_c) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, patch_size * patch_size)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape((B, N, self.patch_size, self.patch_size))
        return x


class MultiScaleMultiHeadTR(nn.Module):
    """
    img_size : the size of feature map extracted by spatial domain
    in_c : the channel of feature map extracted by spatial domain
    n_dim : the embed dimension of the patch embed
    drop_ratio : the dropout rate
    """

    def __init__(self,
                 img_size,
                 in_c,
                 n_dim,
                 drop_ratio):
        super(MultiScaleMultiHeadTR, self).__init__()
        self.patch_size = img_size
        self.in_c = in_c

        num_patches = int((img_size // self.patch_size) ** 2)
        self.pos1 = nn.Parameter(torch.zeros(1, num_patches, n_dim))
        self.eb1 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att1 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        self.pos2 = nn.Parameter(torch.zeros(1, num_patches, n_dim))
        self.eb2 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att2 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        self.pos3 = nn.Parameter(torch.zeros(1, num_patches, n_dim))
        self.eb3 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att3 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        self.pos4 = nn.Parameter(torch.zeros(1, num_patches, n_dim))
        self.eb4 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att4 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)

        self.conv1x1 = nn.Conv2d(4, n_dim, kernel_size=1, stride=1)
        self.pos_drop = nn.Dropout(p=drop_ratio)

    def forward(self, x):
        eb1 = self.eb1(x)
        input1 = self.pos_drop(eb1 + self.pos1)
        att1 = self.att1(input1)

        eb2 = self.eb2(x)
        input2 = self.pos_drop(eb2 + self.pos2)
        att2 = self.att2(input2)
        att2 = att2.reshape(att1.shape)

        eb3 = self.eb3(x)
        input3 = self.pos_drop(eb3 + self.pos3)
        att3 = self.att3(input3)
        att3 = att3.reshape(att1.shape)

        eb4 = self.eb4(x)
        input4 = self.pos_drop(eb4 + self.pos4)
        att4 = self.att4(input4)
        att4 = att4.reshape(att1.shape)

        out = self.conv1x1(torch.cat((att1, att2, att3, att4), dim=1))

        return out


class MMF(nn.Module):
    """
    in_c : the channel of the feature map stacked by spatial domain, frequency domain and MSMHTR
    img_size : the size of the feature map stacked by spatial domain, frequency domain and MSMHTR
    """

    def __init__(self, in_c, img_size=56, r=4):
        super(MMF, self).__init__()
        inter_channels = int(in_c // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_c, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_channels, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
        )

        self.l_convk = nn.Conv2d(in_c, in_c, kernel_size=1, bias=True)
        self.l_convv = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_channels, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
        )

        self.convk = nn.Conv2d(in_c, in_c, kernel_size=1, bias=True)
        self.convv = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.scale = (img_size * img_size * in_c) ** -0.5

        self.conv1x1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=1)

    def forward(self, x_s, x_fq, x_mt):
        # local feature
        l_q = self.local_att(x_s)
        l_k = self.l_convk(x_s)
        l_v = self.l_convv(x_s)

        fuse = (l_q @ l_k.transpose(-2, -1)) * self.scale
        fuse = fuse.softmax(dim=-1)
        l_fuse = fuse @ l_v

        # global feature
        g_q = self.global_att(self.conv1x1(x_mt))
        g_k = self.convk(x_fq)
        g_v = self.convv(x_fq)

        fuse = (g_q * g_k) * self.scale
        fuse = fuse.softmax(dim=-1)
        g_fuse = fuse @ g_v

        f_cmf = self.conv1(l_fuse + g_fuse)

        return f_cmf


class UniM2CT(nn.Module):

    def __init__(self, img_size, n_dim, drop_ratio):
        super(UniM2CT, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b4')
        state_dict = torch.load('xx/pretrain/efficientnet-b4-6ed6700e.pth')
        self.model.load_state_dict(state_dict)
        self.out = {}

        # 上采样层
        self.backbone1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.Conv2d(28, 128, kernel_size=1, stride=1)
        )

        self.ex_fq = ExtraFreq(img_size=img_size, out_c=128)
        self.mt = MultiScaleMultiHeadTR(img_size=int(img_size / 4), in_c=128, n_dim=128, drop_ratio=drop_ratio)

        self.lgf = MMF(in_c=128, img_size=img_size / 4)

        self.backbone2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def feature_forward(self, x):
        return self.backbone1(self.model.extract_features(x))

    def forward(self, x):
        f_s = self.feature_forward(x)  # [B,128,H/4,W/4]
        f_fq = self.ex_fq(x)  # [B,128,H/4,W/4]
        f_mt = self.mt(f_s)  # [B,128,H/4,W/4]

        f_lgf = self.lgf(f_s, f_fq, f_mt)
        feat_out = self.backbone2(f_lgf)
        out = self.fc(feat_out)
        self.out['feat'] = feat_out
        self.out['logits'] = out.squeeze(-1)
        return self.out

def create_model(img_size=224, n_dim=384, drop_ratio=0.1):
    return UniM2CT(img_size=img_size, n_dim=n_dim, drop_ratio=drop_ratio)

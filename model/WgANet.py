import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from model.SwinUMamba import VSSMEncoder, VSSBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import pywt
import re
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def get_wavelet_filters(wavelet_name="haar"):
    """
    根据小波基名称获取分解滤波器和重建滤波器
    wavelet_name: "haar", "db2", "coif1", "sym2" 等
    """
    wavelet = pywt.Wavelet(wavelet_name)
    dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)  # 分解低通
    dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)  # 分解高通
    rec_lo = torch.tensor(wavelet.rec_lo, dtype=torch.float32)        # 重建低通
    rec_hi = torch.tensor(wavelet.rec_hi, dtype=torch.float32)        # 重建高通
    return dec_lo, dec_hi, rec_lo, rec_hi

def pad_if_needed(x, kernel_size):
    """自动 padding，保证输入至少 >= kernel_size"""
    h, w = x.shape[2], x.shape[3]
    pad_h = max(0, kernel_size - h)
    pad_w = max(0, kernel_size - w)
    if pad_h > 0 or pad_w > 0:
        # 左右上下： (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x

def dwt_init(x, wavelet_name="haar"):
    """离散小波变换 (DWT)，支持多小波基"""
    B, C, H, W = x.shape
    wavelet = pywt.Wavelet(wavelet_name)
    dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32, device=x.device)
    dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32, device=x.device)

    # 转换为卷积核
    dec_lo = dec_lo.view(1, 1, -1)
    dec_hi = dec_hi.view(1, 1, -1)

    lo_filter = dec_lo.unsqueeze(3)  # (1,1,k,1)
    hi_filter = dec_hi.unsqueeze(3)
    lo_col_filter = dec_lo.unsqueeze(2)  # (1,1,1,k)
    hi_col_filter = dec_hi.unsqueeze(2)

    # padding，防止 kernel > feature map
    x = pad_if_needed(x, max(len(wavelet.dec_lo), len(wavelet.dec_hi)))

    # 行滤波
    lo_rows = F.conv2d(x, lo_filter.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)
    hi_rows = F.conv2d(x, hi_filter.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)

    # 列滤波
    x_LL = F.conv2d(lo_rows, lo_col_filter.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)
    x_LH = F.conv2d(lo_rows, hi_col_filter.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)
    x_HL = F.conv2d(hi_rows, lo_col_filter.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)
    x_HH = F.conv2d(hi_rows, hi_col_filter.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)

    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x, wavelet_name="haar"):
    """二维 IWT"""
    B, in_channel, H, W = x.shape
    C = in_channel // 4
    _, _, rec_lo, rec_hi = get_wavelet_filters(wavelet_name)

    rec_lo = rec_lo.to(x.device)
    rec_hi = rec_hi.to(x.device)

    # 拆分子带
    x_LL = x[:, 0:C, :, :]
    x_HL = x[:, C:2*C, :, :]
    x_LH = x[:, 2*C:3*C, :, :]
    x_HH = x[:, 3*C:4*C, :, :]

    # 构造卷积核 (行、列两个方向)
    lo_row_filter = rec_lo.view(1,1,-1,1).repeat(C,1,1,1)
    hi_row_filter = rec_hi.view(1,1,-1,1).repeat(C,1,1,1)
    lo_col_filter = rec_lo.view(1,1,1,-1).repeat(C,1,1,1)
    hi_col_filter = rec_hi.view(1,1,1,-1).repeat(C,1,1,1)

    # 列方向逆变换
    lo = F.conv_transpose2d(x_LL, lo_col_filter, stride=(1,2), groups=C) + \
         F.conv_transpose2d(x_LH, hi_col_filter, stride=(1,2), groups=C)

    hi = F.conv_transpose2d(x_HL, lo_col_filter, stride=(1,2), groups=C) + \
         F.conv_transpose2d(x_HH, hi_col_filter, stride=(1,2), groups=C)

    # 行方向逆变换
    out = F.conv_transpose2d(lo, lo_row_filter, stride=(2,1), groups=C) + \
          F.conv_transpose2d(hi, hi_row_filter, stride=(2,1), groups=C)

    return out

class DWT(nn.Module):
    def __init__(self, wavelet_name="haar"):
        super(DWT, self).__init__()
        self.wavelet_name = wavelet_name
        self.requires_grad = False  # 小波变换不需要训练

    def forward(self, x):
        return dwt_init(x, wavelet_name=self.wavelet_name)

class IWT(nn.Module):
    def __init__(self, wavelet_name="haar"):
        super(IWT, self).__init__()
        self.wavelet_name = wavelet_name
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x, wavelet_name=self.wavelet_name)

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height 
        d = max(int(in_channels/reduction),4) 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU()) 

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias)) 
        
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0] 
        n_feats =  inp_feats[0].shape[1] 
        

        inp_feats = torch.cat(inp_feats, dim=1) # b 3c h//2 w//2
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3]) 
        
        feats_U = torch.sum(inp_feats, dim=1) 
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S) 

        attention_vectors = [fc(feats_Z) for fc in self.fcs] 
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1) 
        # stx()
        attention_vectors = self.softmax(attention_vectors) 
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1) 
        
        return feats_V   

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Squeeze操作：全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation操作：全连接层
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Sigmoid激活函数，用于输出注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze操作
        # print('x ',x.shape)
        avg_out = self.global_avg_pool(x)
        # print('avgx ',avg_out.shape)
        
        # Excitation操作
        x = self.fc1(avg_out)
        x = F.relu(x)
        x = self.fc2(x)
        
        # 得到通道注意力权重
        attention = self.sigmoid(x)
        
        # 通道加权
        return x * attention

class Wide_Transformer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Wide_Transformer, self).__init__()
        self.num_heads = num_heads
        dim1 = dim * num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim1 * 4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim1 * 4, dim1 * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=bias)
        self.project_out = nn.Conv2d(dim1, dim1, kernel_size=1, bias=bias)
        self.project_out_1 = nn.Conv2d(dim1, dim, kernel_size=1, bias=bias)
        self.channel=ChannelAttention(dim1)

    def forward(self, x):
        res=x
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v,l = qkv.chunk(4, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # l = rearrange(l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        L=self.channel(l)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out=L+out

        # out = self.project_out(out)+L
        out = self.project_out_1(out)+res
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Wide_Transformer(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x):
       
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class MDAM(nn.Module):
    def __init__(self, out_dim,in_dim):
        super(MDAM, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.tongdao = ConvBNReLU(out_dim,in_dim,1,1)
        self.t = 30
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.concat_conv1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim)
        )
        self.concat_conv2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim) 
        )
        self.linear_layers = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1,bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim , kernel_size=1),
        )
        self.dynamic_aggregation = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 4, 2, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_ful, x_rgb, x_thermal):
        x_ful = self.tongdao(x_ful)
        x_thermal = self.tongdao(x_thermal)

        x_ful_rgb = x_ful.mul(x_rgb)
        x_ful_thermal = x_ful.mul(x_thermal)

        x_concat = self.concat_conv1(torch.cat([x_ful_rgb, x_ful_thermal], dim=1))

        weight = self.avg_pool(x_concat).view(x_concat.size(0), x_concat.size(1))
        weight = self.dynamic_aggregation(weight)
        weight = F.softmax(weight / self.t, dim=1)

        # x_rgb_att = torch.sigmoid(self.linear_layers(x_rgb))
        x_rgb_att = torch.mul(x_concat, x_rgb)

        x_thermal_att = torch.mul(x_concat, x_thermal)

        x_att = x_rgb_att * weight[:, 0].view(x_concat.size(0), 1, 1, 1) + x_thermal_att * weight[:, 1].view(x_concat.size(0), 1, 1, 1)
        out = self.concat_conv2(x_att)
        out = self.relu(out) + x_rgb

        return out

class GlobalLocalContextBlock(nn.Module):
    # def __init__(self, in_dim, out_dim):
    def __init__(self, dim):
        super(GlobalLocalContextBlock, self).__init__()
        in_dim = dim
        out_dim = dim

        # 多尺度空洞卷积提取全局上下文
        self.dconv1 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=3, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=3, padding=2, dilation=2)
        self.dconv4 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=3, padding=4, dilation=4)
        self.dconv8 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=3, padding=8, dilation=8)

        # 融合后的1x1卷积 + BN + ReLU
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 可选：局部路径（常规卷积）
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 融合全局和局部
        self.final_fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 多尺度全局上下文分支
        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv4(x)
        x4 = self.dconv8(x)
        global_feat = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))

        # 局部分支
        local_feat = self.local_conv(x)

        # 融合
        out = self.final_fuse(torch.cat([global_feat, local_feat], dim=1))
        return out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):

        B, C, H, W = x.size()

        pad_w = (ps - W % ps) % ps
        pad_h = (ps - H % ps) % ps

        if pad_w == 0 and pad_h == 0:
            return x
        use_reflect = (pad_w < W) and (pad_h < H)

        if use_reflect:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return x


    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)
        # print("x shape before pad:", x.shape)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class WgANet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                #  backbone_name='swsl_resnet50',
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm,
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.act1 = self.backbone.act1
        self.maxpool = self.backbone.maxpool
        self.layers = nn.ModuleList()
        self.layers.append(self.backbone.layer1)
        self.layers.append(self.backbone.layer2)
        self.layers.append(self.backbone.layer3)
        self.layers.append(self.backbone.layer4)

        self.ful_layer3 = MDAM(768,encoder_channels[3])
        self.ful_layer2 = MDAM(384,encoder_channels[2])
        self.ful_layer1 = MDAM(192,encoder_channels[1])
        self.ful_layer0 = MDAM(96,encoder_channels[0])

        self.stem = nn.Sequential(
              nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3),
              nn.InstanceNorm2d(48, eps=1e-5, affine=True),
        )
        self.down1 = nn.Conv2d(48, 96, kernel_size=2, stride=2)
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=48)

        ssm_dims=[96, 192, 384, 768]

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)


        base_d_state = 16
        ## 1
        self.Fmamba1 = VSSBlock(hidden_dim=ssm_dims[0],
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=base_d_state)        
        self.DTB1 = DTB(dim=ssm_dims[0], num_heads=heads[0], ffn_factor=ffn_factor, 
                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.GlobalLocalContextBlock1 = GlobalLocalContextBlock(dim=ssm_dims[0])
        self.h_out_conv1 = nn.Conv2d(ssm_dims[0], ssm_dims[0]*3, 3, 1, 1)
        self.h_fusion1 = SKFF(ssm_dims[0], height=3, reduction=8)
        ## 2
        self.down2 = nn.Conv2d(ssm_dims[0], ssm_dims[1], kernel_size=2, stride=2)
        self.Fmamba2 = VSSBlock(hidden_dim=ssm_dims[1],
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=base_d_state)        
        self.DTB2 = DTB(dim=ssm_dims[1], num_heads=heads[1], ffn_factor=ffn_factor, 
                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.GlobalLocalContextBlock2 = GlobalLocalContextBlock(dim=ssm_dims[1])
        self.h_out_conv2 = nn.Conv2d(ssm_dims[1], ssm_dims[1]*3, 3, 1, 1)
        self.h_fusion2 = SKFF(ssm_dims[1], height=3, reduction=8)
        ## 3
        self.down3 = nn.Conv2d(ssm_dims[1], ssm_dims[2], kernel_size=2, stride=2)
        self.Fmamba3 = VSSBlock(hidden_dim=ssm_dims[2],
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=base_d_state)        
        self.DTB3 = DTB(dim=ssm_dims[2], num_heads=heads[2], ffn_factor=ffn_factor, 
                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.GlobalLocalContextBlock3 = GlobalLocalContextBlock(dim=ssm_dims[2])
        self.h_out_conv3 = nn.Conv2d(ssm_dims[2], ssm_dims[2]*3, 3, 1, 1)
        self.h_fusion3 = SKFF(ssm_dims[2], height=3, reduction=8)
        ## 4
        self.down4 = nn.Conv2d(ssm_dims[2], ssm_dims[3], kernel_size=2, stride=2)
        self.Fmamba4 = VSSBlock(hidden_dim=ssm_dims[3],
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=base_d_state)        
        self.DTB4 = DTB(dim=ssm_dims[3], num_heads=heads[3], ffn_factor=ffn_factor, 
                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.GlobalLocalContextBlock4 = GlobalLocalContextBlock(dim=ssm_dims[3])
        self.h_out_conv4 = nn.Conv2d(ssm_dims[3], ssm_dims[3]*3, 3, 1, 1)
        self.h_fusion4 = SKFF(ssm_dims[3], height=3, reduction=8)
    def forward(self, x):
        n, c, h, w = x.size()
        ## 空间mamba
        ssmx = self.stem(x)
        vss_outs = self.vssm_encoder(ssmx) # 48*128*128, 96*64*64, 192*32*32, 384*16*16, 768*8*8

        ## 小波基
        dwt = DWT("haar")
        idwt = IWT("haar")
        ## Daubechies (db2)
        # dwt = DWT("db2")
        # idwt = IWT("db2")
        ## Coiflet (coif1)
        # dwt = DWT("coif1")
        # idwt = IWT("coif1")
        # ## Symlet (sym2)
        # dwt = DWT("sym2")
        # idwt = IWT("sym2")
        
        #### 频率1
        y = self.stem(x) # n 48 h//2 w//2
        y = self.down1(y) # n 96 h//4 w//4
        input_LL1, x_HL, x_LH, x_HH = dwt(y) # n 96 h//8 w//8
        ## 低频
        input_LL1 = input_LL1.permute(0, 2, 3, 1) # n h//8 w//8 96
        input_LL1 = self.Fmamba1(input_LL1).permute(0, 3, 1, 2) # n 96 h//8 w//8
        # print('input_LL1 ',input_LL1.shape)
        ## 高频
        y_h1 = self.h_fusion1([x_HL, x_LH, x_HH]) # n 96 h//8 w//8
        y_h1 = self.DTB1(y_h1)
        # y_h1 = self.GlobalLocalContextBlock1(y_h1)
        y_h1 = self.h_out_conv1(y_h1) # n 3*96 h//8 w//8
        # print('y_h1 ',y_h1.shape)

        y_1 = idwt(torch.cat([input_LL1, y_h1], dim=1)) # n 96 h//4 w//4
        # print('y_1 ',y_1.shape)


        #### 频率2
        y = self.down2(y_1) 
        input_LL2, x_HL, x_LH, x_HH = dwt(y)

        input_LL2 = input_LL2.permute(0, 2, 3, 1)
        input_LL2 = self.Fmamba2(input_LL2).permute(0, 3, 1, 2) 

        y_h2 = self.h_fusion2([x_HL, x_LH, x_HH])
        # print('y_h2 ',y_h2.shape)
        y_h2 = self.DTB2(y_h2)
        # y_h2 = self.GlobalLocalContextBlock2(y_h2)
        y_h2 = self.h_out_conv2(y_h2) 

        y_2 = idwt(torch.cat([input_LL2, y_h2], dim=1)) 
        # print('y_2 ',y_2.shape)


        #### 频率3
        y = self.down3(y_2)
        input_LL3, x_HL, x_LH, x_HH = dwt(y)

        input_LL3 = input_LL3.permute(0, 2, 3, 1) 
        input_LL3 = self.Fmamba3(input_LL3).permute(0, 3, 1, 2) 

        y_h3 = self.h_fusion3([x_HL, x_LH, x_HH]) 
        y_h3 = self.DTB3(y_h3)
        # y_h3 = self.GlobalLocalContextBlock3(y_h3)
        y_h3 = self.h_out_conv3(y_h3) 

        y_3 = idwt(torch.cat([input_LL3, y_h3], dim=1)) 
        # print('y_3 ',y_3.shape)

        #### 频率4
        y = self.down4(y_3) 
        input_LL4, x_HL, x_LH, x_HH = dwt(y)

        input_LL4 = input_LL4.permute(0, 2, 3, 1) 
        input_LL4 = self.Fmamba4(input_LL4).permute(0, 3, 1, 2) 

        y_h4 = self.h_fusion4([x_HL, x_LH, x_HH]) 
        y_h4 = self.DTB4(y_h4)
        # y_h4 = self.GlobalLocalContextBlock4(y_h4)
        y_h4 = self.h_out_conv4(y_h4) 

        y_4 = idwt(torch.cat([input_LL4, y_h4], dim=1)) 
        # print('y_4 ',y_4.shape)

        ## CNN
        ress = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x1 = self.layers[0](x)
        # print('x1 ',x1.shape)
        # print('vss_outs[1] ',vss_outs[1].shape)
        ful_1 = self.ful_layer0(y_1, x1, vss_outs[1]) 


        x2 = self.layers[1](ful_1)
        ful_2 = self.ful_layer1(y_2, x2, vss_outs[2])


        x3 = self.layers[2](ful_2)
        ful_3 = self.ful_layer2(y_3, x3, vss_outs[3])


        x4 = self.layers[3](ful_3)
        ful_4 = self.ful_layer3(y_4, x4, vss_outs[4])


        x = self.decoder(ful_1, ful_2, ful_3, ful_4, h, w)
        return x

def load_pretrained_ckpt(
    model, 
    ckpt_path="/private/workspace/WAFAAG3/pretrain/vmamba_tiny_e292.pth"
):

    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias", 
                   "patch_embed.proj.weight", "patch_embed.proj.bias", 
                   "patch_embed.norm.weight", "patch_embed.norm.weight"]    

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")
        
    model.load_state_dict(model_dict)
    print('Load vmamba_tiny_e292 Done!')

    return model

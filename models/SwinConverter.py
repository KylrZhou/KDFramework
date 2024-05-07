from utils import MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(K.size(-1))), dim=-1)
        out = attention_weights @ V
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, output_dim, output_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.conv = nn.Conv2d(dim, output_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(x.size(0), -1, int(x.size(1)**0.5), int(x.size(1)**0.5))
        x = self.conv(x)
        x = self.pool(x)
        return x

class Reformer(nn.Module):
    def __init__(self,in_channels,  stage=3):
        super().__init__()
        if stage == 4:
            self.reform_conv = nn.Conv2d(in_channels=in_channels, out_channels=2048, kernel_size=1, stride=1, padding=0)
        elif stage == 3:
            self.reform_conv = nn.Conv2d(in_channels=in_channels, out_channels=2048, kernel_size=1, stride=1, padding=0)
        elif stage == 2:
            self.reform_conv = nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=1, stride=1, padding=0)
        elif stage == 1:
            self.reform_conv = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        _, _, h, w = y.size()
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.reform_conv(x)
        return x/2 + y/2

@MODEL.register()
class SwinConverter(nn.Module):
    def __init__(self, in_channels, stage=3):
        super().__init__()
        if stage == 1:
            self.reform_conv = Reformer(in_channels, 1)
            self.attn = nn.Sequential(
                #SwinTransformerBlock(16, 512, 512, 16)
                SwinTransformerBlock(in_channels, 128, 512, 16)
            )
        
        elif stage == 2:
            self.reform_conv = Reformer(in_channels, 2)
            self.attn = nn.Sequential(
                #SwinTransformerBlock(32, 512, 1024, 8),
                SwinTransformerBlock(in_channels, 256, 1024, 8),
                #SelfAttention(1024)
            )
        
        elif stage == 3:
            self.reform_conv = Reformer(in_channels, 3)
            self.attn = nn.Sequential(
                SwinTransformerBlock(in_channels, 768, 1024, 4),
                SwinTransformerBlock(1024, 1536, 2048, 4),
            )
        elif stage == 4:
            self.reform_conv = Reformer(in_channels, 4)
            self.attn = nn.Sequential(
                SwinTransformerBlock(in_channels, 1664, 2048, 4),
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        _ = x
        x = self.attn(x)
        x = self.reform_conv(_, x)
        return x
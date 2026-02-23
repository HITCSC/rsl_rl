import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128, target_size=(64, 64)):
        super(CNNEncoder, self).__init__()
        
        # 目标统一尺寸
        self.target_size = target_size
        self.embedding_dim = embedding_dim
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ELU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ELU(inplace=True)
            )

        # 网络结构
        self.layer1 = conv_block(in_channels, 16)
        self.layer2 = conv_block(16, 32)
        self.layer3 = conv_block(32, 64)
        self.layer4 = conv_block(64, 128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim), 
            nn.Tanh()
        )

    def forward(self, x):
        # --- 输入端标准化 (Input Resizing) ---
        # 无论原始尺寸是 (17, 11) 还是 (161, 101)
        # 统一缩放到 target_size，保证卷积核看到的“视野”比例一致
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # 特征提取
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        # x = self.layer3(x)
        x = self.pool(self.layer3(x))
        x = self.layer4(x)
        
        # 降维对齐
        x = self.adaptive_pool(x)
        output = self.fc(x)
        
        return output

# class SmallCNNEncoder(nn.Module):
#     def __init__(
#             self, 
#             embedding_dim=128,
#     ):
#         super(SmallCNNEncoder, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * 7 * 13, embedding_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.cnn(x)


# class LargeCNNEncoder(nn.Module):
#     def __init__(
#             self, 
#             embedding_dim=512,
#     ):
#         super(LargeCNNEncoder, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=9, stride=2), nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=9, stride=2), nn.ReLU(),
#             nn.AdaptiveAvgPool2d((16, 16)), 
#             nn.Flatten(),
#             nn.Linear(32 * 16 * 16, embedding_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.cnn(x)


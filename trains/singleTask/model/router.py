#Router:根据输入得到重要性权重：输入（2 张图片的特征：(2,64,8,8)）→ 展平成向量（(2,4096)）→ 降维（(2,512)）→ 归一化 + ReLU → 映射到 4 个专家（(2,4)）→ 温度调节 → 输出 4 个专家的权重概率（(2,4)）。
import torch
import torch.nn as nn
import torch.nn.functional as F

class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        self.t = t
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1)))/self.t
        output = torch.softmax(x, dim=1)
        return output

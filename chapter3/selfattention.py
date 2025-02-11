import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        简易自注意力层
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        """
        super(SelfAttention, self).__init__()

        # 定义线性层，用于生成 Q、K、V
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # 输出变换
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        :param x: 输入张量，形状 [batch_size, seq_len, hidden_dim]
        :output: 自注意力计算后的输出，形状 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 线性变换得到 Q, K, V
        # 形状: [batch_size, seq_len, hidden_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数: Q @ K^T / sqrt(hidden_dim)
        # Q: [batch_size, seq_len, hidden_dim]
        # K^T: [batch_size, hidden_dim, seq_len]
        # scores: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)

        # 通过 softmax 得到注意力分布
        attn_weights = torch.softmax(scores, dim=-1)

        # 注意力加权 V
        # [batch_size, seq_len, seq_len] x [batch_size, seq_len, hidden_dim]
        # => [batch_size, seq_len, hidden_dim]
        attn_output = torch.matmul(attn_weights, V)

        # 输出层
        output = self.out(attn_output)

        return output

if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 5
    input_dim = 256
    hidden_dim = 256
    x = torch.randn(batch_size, seq_len, input_dim)

    attention = SelfAttention(input_dim, hidden_dim)
    out = attention(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
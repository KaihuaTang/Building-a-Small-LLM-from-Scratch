import torch
import torch.nn as nn
import math

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        """
        简易自注意力层
        :param embed_dim: 输入/输出的特征维度
        :param num_heads: 多头注意力的头数，这里可以指定为8表示单头注意力
        """
        super(MultiHeadCausalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 将嵌入维度平分到各个头上
        # 注意实际应用中需要确保 embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # 定义线性层，用于生成 Q、K、V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 输出变换
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        :param x: 输入张量，形状 [batch_size, seq_len, embed_dim]
        :param mask: 可选的掩码（mask），形状与注意力矩阵匹配，如 [batch_size, 1, seq_len, seq_len]
        :return: 自注意力计算后的输出，形状 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # 线性变换得到 Q, K, V
        # 形状: [batch_size, seq_len, embed_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 多头展开
        # 变换后形状: [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 将 [batch_size, seq_len, num_heads, head_dim] 转成 [batch_size, num_heads, seq_len, head_dim]
        Q = Q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        # Q: [batch_size, num_heads, seq_len, head_dim]
        # K^T: [batch_size, num_heads, head_dim, seq_len]
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果有 mask，则在计算分数时将被mask的部分赋予一个很大的负数，以避免注意力
        # mask 形状一般为 [batch_size, 1, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            scores = scores + mask

        # 通过 softmax 得到注意力分布
        attn_weights = torch.softmax(scores, dim=-1)
        print("注意力权重分布:", attn_weights)

        # 注意力加权 V
        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # => [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # 把多头重新拼接回原始形状
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        # 拼接头部维度
        # => [batch_size, seq_len, embed_dim]
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # 输出层
        output = self.out(attn_output)

        return output

if __name__ == "__main__":
    # 测试代码
    batch_size = 1
    seq_len = 5
    embed_dim = 256
    num_heads = 2
    x = torch.randn(batch_size, seq_len, embed_dim)
    causal_mask = torch.triu(torch.ones(5, 5, device=x.device, dtype=torch.float),
                            diagonal=1
                        )
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
    
    attention = MultiHeadCausalSelfAttention(embed_dim, num_heads=num_heads)
    out = attention(x, mask=causal_mask)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)

# 03 DeepSeekV3的注意力优化

申明：本教程的所有内容(文字，图片，代码等)可以用于非盈利目的个人使用和分享。但如果用于盈利目的，包括但不限于卖课，公众号，视频号等需要经由作者的批准。谢谢理解。[\[知乎链接\]](https://zhuanlan.zhihu.com/p/19275166926)

[\[主目录链接\]](https://github.com/KaihuaTang/All-you-need-to-know-about-LLM#章节链接)


## 前言
书接上文[注意力模块与KV Cache](chapter2/README.md)，介绍完了基本的大语言模型的注意力模块与相关的KV-Cache，我们本章着重展开讲讲主流开源大模型注意力模块的后续改良与优化。本章我们不仅会介绍最近当红的[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)注意力优化原理，更会直接深扒一下Qwen2/LLaMA3和DeepSeekV3具体的注意力模块的代码，深入讲解每一行代码对应的功能和原理。(DeepSeek太火了，被拉回国项目攻关了，更新耽误了两周)

## 一. 注意力优化
主流的开源大模型网络的注意力计算机制除了上一章介绍的多头注意力Multi-Head Attention(MHA)以外，最近也有了新的变种，主要包括Multi-Query Attention (MQA)，Grouped-Query Attention (GQA)和最近当红的DeepSeek的Multi-head Latent Attention (MLA)而他们优化的方向其实是一致的，就是极致的压缩KV的大小，因为这样KV-Cache可以加载的更快。毕竟现在都在说超长上下文，token数N长了KV-Cache优化后加载KV的传输带宽开销可也不小啊。

<div align="center">
    <img src="03-1.png" alt="logo" width="100%"  style="padding-bottom: 20px"/>
    图1：各种注意力优化方案。
</div>

### 1. Multi-Query Attention (MQA) / 多查询注意力

参考上一章的标准多头注意力Multi-Head Attention(MHA)的代码，其QKV构成如下：
```
# x为attention模块输入的所有token的hidden states
batch_size, seq_len, input_dim = x.shape

# 线性变换得到 Q, K, V
Q = self.query(x)    
K = self.key(x)
V = self.value(x)

# 多头展开
# 变换后形状: [batch_size, seq_len, num_heads, head_dim]
Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

# 此处省略后续attention计算
```
其中K和V对应的线性层self.key和self.value都必须将token的特征从input_dim维度映射到num_heads * head_dim维度。也就是说这两层的线性层权重的张量形状为[num_heads * head_dim, input_dim]。而同时KV Cache也必须存储两个[batch_size, seq_len, num_heads, head_dim]大小的张量。

而多查询注意力Multi-Query Attention (MQA)做的优化直白来来讲就是只对Query做多头注意力，而Key和Query只生成单头。此时self.key和self.value线性层权重的张量形状为[head_dim, input_dim]，这不仅让线性层计算量缩小num_head倍，KV Cache的大小也缩小了同样的倍数，计算量和带宽双收益。当然天下没有免费的午餐，由于Key和Value仅有一个头，其表达能力肯定是有所损失的。具体MQA参考代码如下：

```
# x为attention模块输入的所有token的hidden states
batch_size, seq_len, input_dim = x.shape

# 线性变换得到 Q, K, V
Q = self.query(x)
K = self.key(x)        # 注意此处的key线性层输出维度小于MHA
V = self.value(x)      # 注意此处的key线性层输出维度小于MHA

# 多头展开
Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
K = K.view(batch_size, seq_len, 1, self.head_dim)
V = V.view(batch_size, seq_len, 1, self.head_dim)

# 将K和V复制若干份，扩展至: [batch_size, seq_len, num_heads, head_dim]
K = K.repeat(1, 1, self.num_heads, 1)
V = V.repeat(1, 1, self.num_heads, 1)

# 此处省略后续attention计算
```


### 2. Grouped-Query Attention (GQA) / 组查询注意力

正如上文说的，MQA必然带来注意力层表达能力的下降。因此就有了组查询注意力Grouped-Query Attention (GQA)，这其实是MHA和MQA的折中，就是Query保留所有head数的情况下，Key和Value不止一个头，而是保留num_group数的头(num_groups <= num_heads且num_heads可以被num_groups整除)，我们不难发现GQA是MHA和MQA的一种泛化形式，num_groups=1时就是MQA，num_groups=num_heads时就是MHA。可以说是万金油的表达形式。因为GQA是更泛化的表达形式，同时也有个额外的参数num_groups（有的代码中也叫num_key_value_heads）可以调，因此GQA往往可以调到与MHA差不多的性能，同时又能有KV Cache和线性层计算减少的收益。在主流的Qwen2和LLaMA3的代码中，一般也都支持GQA的配置。

具体GQA参考代码如下：
```
# x为attention模块输入的所有token的hidden states
batch_size, seq_len, input_dim = x.shape

# 线性变换得到 Q, K, V
Q = self.query(x)
K = self.key(x)        
V = self.value(x)      

# 多头展开
# num_groups <= num_heads且num_heads可以被num_groups整除
Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
K = K.view(batch_size, seq_len, 1, self.num_groups, self.head_dim)   
V = V.view(batch_size, seq_len, 1, self.num_groups, self.head_dim)

# 将K和V复制若干份，扩展至: [batch_size, seq_len, num_heads, head_dim]
K = K.repeat(1, 1, self.num_heads // self.num_groups, 1, 1).view(batch_size, seq_len, self.num_heads, self.head_dim)
V = V.repeat(1, 1, self.num_heads // self.num_groups, 1, 1).view(batch_size, seq_len, self.num_heads, self.head_dim)

# 此处省略后续attention计算
```

### 3. Multi-head Latent Attention (MLA) / 多头潜在注意力

最近大火的[DeepSeek系列(从V2到V3)](https://github.com/deepseek-ai/DeepSeek-V3/tree/main)则采用了一种比GQA更极致的压缩方式，不仅进一步减少了注意力层中线性层的理论算力，更把KV Cache压缩到了新的境界。在介绍MLA之前，先介绍一个低秩分解在大模型上应用的例子，后面我们可能也会单独详细讲讲，就是[LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)。这是大模型常用的一种高性能微调方法，其具体概念就是，如果线性层的权重$W$太大(假设其张量形状为[out_dim, in_dim])，训练这个权重太耗显存了，我们可以训练两个更小的权重$W_a$和$W_b$(形状分别为[K, in_dim]和[out_dim, K]，K << in_dim, K << out_dim)。由于K远远小于in_dim和out_dim，这两个权重加起来也远远小于原始的$W$。参考概念如下图2。

<div align="center">
    <img src="03-2.png" alt="logo" width="100%"  style="padding-bottom: 20px"/>
    图2：LoRA微调概念图。
</div>

当我第一次看到DeepSeek的多头潜在注意力Multi-head Latent Attention (MLA)，我首先映入脑袋的便是LoRA，区别是在MLA中并不是额外学两个小的线性权重，而是用直接用两个小的线性权重取代一个完整的线性层。具体MLA的网络结构如下图3（注意MLA在DeepSeek中有两种形式，一种是只有KV的线性层运用了低秩分解，一种是Q和KV都利用了低秩分解，后者也是最近大火的[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/tree/main)和[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/tree/main)的网络结构，因此我们这里以后者为例）。

<div align="center">
    <img src="03-3.png" alt="logo" width="100%"  style="padding-bottom: 20px"/>
    图3：多头潜在注意力结构，参数规模参考DeepSeek-V3与R1。
</div>

上图是MLA如何生成Query，Key和Value的流程图。MLA结构中Query和Key都分为nope部分和rope部分，前者是指No Position Embedding即无需位置编码，后者指Rotary Position Embedding即需要旋转位置编码，而旋转位置编码则是图上apply rotary pos embed模块，该模块主要为了给token提供其在序列中与其他token的相对位置信息，下一章我们会展开详细讲解。

我们可以发现，原始的注意力仅需简单的三个Linear加上Reshape就可以生成Query，Key和Value，但MLA似乎让网络变得更复杂了。这是为什么呢？因为MLA利用了低秩分解的概念，将一个大的矩阵乘拆解为两个小的矩阵乘加一个归一化层。我们可以通过参数量估算发现，如果用三个线性层直接生成同样的Query, Key, Value维度，参数量需要扩大6.7倍。

除了参数量的降低，MLA更可以大幅降低KV Cache，如上图DeepSeek-V3与R1网络中，其每个token仅需保留64维的k_pe + 右侧RMSNorm后的512维特征，总计576维每token的Cache即可。因为其余的k_nope和Value都可以直接通过512维的特征再经过一个线性层得到。而此前其他大语言模型每个token需要多少维度的特征呢？以[72B的Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)模型为例，即便已经使用了组查询注意力GQA，每个token依然需要2048维(128 x 8 x 2)。DeepSeek的MLA将KV Cache压缩了3.5倍，当然这里存的已经不是标准的Key和Value了，需要引入额外的线性层计算才能转换为Key和Value，这就涉及到更复杂的算力和带宽的权衡了。


## 二. 大语言模型中的注意力
下面，让我们切切实实的看一看真实的大模型网络结构里注意力都长什么样。下面我会拿Qwen2/LLaMA3（这两个网络的注意力部分非常相似，因此我仅展示一个）与DeepSeekV3的实际代码进行演示。我会尽可能保留原始代码，仅出于可读性做一些修改，然后通过详细的注释来阐释每一块的作用。

### 1. Qwen2/LLaMA3的注意力代码详解

以transformers库v4.49.0版的代码为例，我们看一下Qwen2注意力模块的实现（LLaMA与他也是大同小异）。完整代码参考链接：[https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2/modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2/modeling_qwen2.py)

```
class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config   # 获取配置参数
        self.layer_idx = layer_idx    # 获取当前是第几层网络
        # 获取注意力头每个头的维度，如果不直接提供head_dim参数则通过hidden_size除以注意力头数获得。
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # 这里的GQA的num_key_value_groups是要复制几遍KV的意思，我上文说的group则是kv本身有几个头。
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
  
        self.scaling = self.head_dim**-0.5   # attention计算时的缩放参数
        self.attention_dropout = config.attention_dropout    # attention是否开启dropout
        self.is_causal = True  # 是否是因果注意力
        # 一些线性层的初始化
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 标准的通过线性层生成query, key和value。 hidden_shape将头数的维度设为动态的。
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # 下面两行为位置编码相关，后续章节会详细讲
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 下面部分为更新kv cache
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # sliding window是一种特殊的注意力稀疏机制，可以减少计算量增加推理长度，比较进阶。可以忽略下面sliding window相关。
        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        # attention的计算有很多优化库，比如flash attention等。attention_interface决定了调用那种实现。
        # 为了便于讲解，我们默认attention_interface = eager_attention_forward，也就是下面的pytorch实现。
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 调用eager_attention_forward，计算attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        
        # 通过reshape和输出线性层o_proj, 得到注意力层的最后输出结果。
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# 如果涉及上述说到的MQA和GQA，就需要这里的repeat_kv函数，将key和value复制多份，直到与query的头数相等。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # 在MQA和GQA中，将key和value复制多份，直到与query的头数相等。
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 下面就是标准的多头因果注意力，和我们上一章讲解的大同小异。
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attention的droupout实现，会丢掉一些attention map中的值。
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```



### 2. DeepSeekV3的注意力代码详解

DeepSeek-V3的网络结构在写本文时还没有集成如transformers库，所以我们参考的是他官方checkpoint中的网络结构源码：[https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)

```
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config  # 获取配置参数
        self.layer_idx = layer_idx  # 获取当前第几层
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 提取配置文件中的各种参数，与Qwen的大同小异。
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        self.max_position_embeddings = config.max_position_embeddings    # 最大位置编码
        self.rope_theta = config.rope_theta   # 位置编码相关参数

        # 下面的参数就是和Qwen/LLaMA不同的部分了，因为生成QKV的Linear被拆成了两个小Linear + 归一化层的形式，所以需要额外的中间维度参数。
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
        
        # 如果不设置q_lora_rank参数，则query还是通过一个线性层直接生成。
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        # DeepSeek-V3与DeepSeek-R1设置了q_lora_rank参数，因此变成两个Linear + 一个RMSNorm的形式，节省了计算量
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        # Key和Value被作为一个整体通过两个Linear + 一个RMSNorm后再split成key和Value
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        # 正常的输出线性层
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)

        # 位置编码相关，暂时跳过。
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    # 初始化位置编码，包含多种位置编码实现，此处跳过。
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # reshape函数
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        # 通过标准的Linear或我们说的LoRA形式生成query
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # 将query每个头的特征维度分成需要位置编码的部分和不需要的部分
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # 先通过一个线性层，生成压缩后的kv特征：512+64。
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # 将其中key的需要位置编码部分的64维单独拿出来
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # 生成完整的key和value
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        # 将key不需要位置编码的部分，和value部分分离
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        # 通过kvcache状态判断key和value的长度
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # query和key的旋转位置编码部分
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # 分别融合query和key的 无位置编码nope和选择位置编码pe
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        # 更新kv cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
       
        # 后续都是正常的attention计算。
        # mla比较特别的就是query，key和value的维度数不一定一致。
        # 因为只要query和key对齐就可以进行attention了，value其实不需要对齐每个头的维度，仅需要对齐长度即可。
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
```


-------------

[\[主目录链接\]](https://github.com/KaihuaTang/All-you-need-to-know-about-LLM#章节链接)



## 引用链接

```
@misc{tang2025all,
title = {Building a Small LLM from Scratch: a tutorial},
author = {Tang, Kaihua and Zhang, Huaizheng},
year = {2025},
note = {\url{https://github.com/KaihuaTang/Building-a-Small-LLM-from-Scratch}},
}
```

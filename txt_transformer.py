# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

class AdaptiveNorm(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)  # 标准归一化
        self.gamma = nn.Linear(hidden_dim, hidden_dim)   # 缩放参数
        self.beta = nn.Linear(hidden_dim, hidden_dim)    # 偏移参数

    def forward(self, x, cond):
        # print(f"AdaptiveNorm forward called with x shape: {x.shape}, cond shape: {cond.shape}")
        
        # 获取 batch_size 和 seq_len
        batch_size, seq_len, cond_dim = cond.shape
        
        # 将 cond 展平为二维张量
        cond = cond.view(-1, cond_dim)  # (batch_size * seq_len, cond_dim)
        
        # 计算 gamma 和 beta
        gamma = self.gamma(cond).view(batch_size, seq_len, -1)  # 恢复形状为 (batch_size, seq_len, hidden_dim)
        beta = self.beta(cond).view(batch_size, seq_len, -1)    # 恢复形状为 (batch_size, seq_len, hidden_dim)
        
        # 对 x 进行归一化
        x = self.norm(x)  # (batch_size, seq_len, hidden_dim)
        
        # print(f"x shape: {x.shape}, gamma shape: {gamma.shape}, beta shape: {beta.shape}")
        
        # 返回自适应归一化结果
        return gamma * x + beta

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
        assert not torch.isnan(out).any(), "NaN detected in FeedForward output"
        assert not torch.isinf(out).any(), "Inf detected in FeedForward output"
        # print(f"FeedForward output stats: mean={out.mean().item()}, std={out.std().item()}")
        return out

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout,
            batch_first=True, bias=True
        )

    def forward(self, x, cond=None, attention_mask=None):
        assert not torch.isnan(x).any(), "Input x to Attention contains NaN values"
        assert not torch.isinf(x).any(), "Input x to Attention contains Inf values"

        if cond is not None:
            # 使用 cond 作为 key 和 value
            attn_output = self.mha(
                x, cond, cond,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )[0]
        else:
            # 自注意力
            attn_output = self.mha(
                x, x, x,
                key_padding_mask= None
            )[0]
        assert not torch.isnan(attn_output).any(), "NaN detected in attention output"
        assert not torch.isinf(attn_output).any(), "Inf detected in attention output"
        #print(f"Attention output stats: mean={attn_output.mean().item()}, std={attn_output.std().item()}")
        return attn_output


class NormLayer(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(x_dim, eps=1e-6)

    def forward(self, x):
        x = self.norm_final(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_dim, cond_dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim),
                                 nn.LayerNorm(hidden_dim))

        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = Attention(hidden_dim, heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = Attention(hidden_dim, heads, dropout=dropout)

        self.ln3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ff = FeedForward(hidden_dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, attention_mask=None):
        cond = self.mlp(cond)
        x = x + self.self_attn(self.ln1(x), attention_mask=attention_mask)
        if attention_mask is not None and (~attention_mask.bool()).all(dim=-1).any():
            # 对于被全 mask 的样本，直接返回输入（跳过 self_attn 和 cross_attn）
            print("Some samples are fully masked — skipping attention")
        else:
            x = x + self.cross_attn(self.ln2(x), cond, attention_mask=attention_mask)
        x = x + self.ff(self.ln3(x))
        return x
# class Block(nn.Module):
#     def __init__(self, hidden_dim, cond_dim, heads, mlp_dim, dropout=0.):
#         super().__init__()
#         self.adaptive_norm1 = AdaptiveNorm(hidden_dim, cond_dim)
#         self.self_attn = Attention(hidden_dim, heads, dropout=dropout)

#         self.adaptive_norm2 = AdaptiveNorm(hidden_dim, cond_dim)
#         self.cross_attn = Attention(hidden_dim, heads, dropout=dropout)

#         self.adaptive_norm3 = AdaptiveNorm(hidden_dim, cond_dim)
#         self.ff = FeedForward(hidden_dim, mlp_dim, dropout=dropout)

#     def forward(self, x, cond, attention_mask=None):
#         x = x + self.self_attn(self.adaptive_norm1(x, cond), attention_mask=attention_mask)
#         x = x + self.cross_attn(self.adaptive_norm2(x, cond), cond, attention_mask=attention_mask)
#         x = x + self.ff(self.adaptive_norm3(x, cond))
#         return x
    # 全局池化交叉注意力
    # def forward(self, x, cond, attention_mask=None):
    #     # 自注意力
    #     x = x + self.self_attn(self.adaptive_norm1(x, cond), attention_mask=attention_mask)

    #     # 全局池化文本特征
    #     pooled_cond = cond.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]

    #     # 交叉注意力
    #     x = x + self.cross_attn(self.adaptive_norm2(x, cond), pooled_cond, attention_mask=None)

    #     # 前馈网络
    #     x = x + self.ff(self.adaptive_norm3(x, cond))
    #     return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, cond_dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer in range(depth):
            self.layers.append(
                Block(hidden_dim, cond_dim, heads, mlp_dim, dropout=dropout)
            )

    def forward(self, x, cond, attention_mask=None):
        for i,block in enumerate(self.layers):
            x = block(x, cond, attention_mask=attention_mask)
            assert not torch.isnan(x).any(), f"NaN detected in Transformer block {i} output"
            assert not torch.isinf(x).any(), f"Inf detected in Transformer block {i} output"
            # print(f"Block {i} output stats: mean={x.mean().item()}, std={x.std().item()}")
        return x

class Transformer(nn.Module):
    def __init__(self, input_size=16, c=768, hidden_dim=768, cond_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., proj=1):
        super().__init__()

        self.c = c
        self.proj = proj
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size

        self.tok_emb = nn.Embedding(codebook_size+1, c)
        self.pos_emb = nn.Embedding(self.input_size ** 2, c)

        self.first_norm = nn.LayerNorm(c, eps=1e-6)
        self.in_proj = nn.Conv2d(c, hidden_dim, kernel_size=proj, stride=proj, bias=False)

        self.transformer = TransformerEncoder(hidden_dim=hidden_dim, cond_dim=cond_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.out_proj = nn.Sequential(nn.Conv2d(hidden_dim, c*(proj**2), kernel_size=1, stride=1, bias=False),
                                      nn.PixelShuffle(proj)
                                      )

        self.last_norm = nn.LayerNorm(c, eps=1e-6)
        self.bias = nn.Parameter(torch.zeros(1, 1, codebook_size+1))
        
        # 添加text降维
        # self.text_proj = nn.Linear(1024, hidden_dim)  # 将 text 的维度从 1024 降到 hidden_dim

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Init projection in
        nn.init.xavier_uniform_(self.in_proj.weight)

        # Zero-out cross attn projection layers in blocks:
        for block in self.transformer.layers:
                        # 初始化自适应归一化中的 gamma 和 beta
            # nn.init.constant_(block.adaptive_norm1.gamma.weight, 1.0)
            # nn.init.constant_(block.adaptive_norm1.gamma.bias, 0.0)
            # nn.init.constant_(block.adaptive_norm1.beta.weight, 0.0)
            # nn.init.constant_(block.adaptive_norm1.beta.bias, 0.0)

            # nn.init.constant_(block.adaptive_norm2.gamma.weight, 1.0)
            # nn.init.constant_(block.adaptive_norm2.gamma.bias, 0.0)
            # nn.init.constant_(block.adaptive_norm2.beta.weight, 0.0)
            # nn.init.constant_(block.adaptive_norm2.beta.bias, 0.0)

            # nn.init.constant_(block.adaptive_norm3.gamma.weight, 1.0)
            # nn.init.constant_(block.adaptive_norm3.gamma.bias, 0.0)
            # nn.init.constant_(block.adaptive_norm3.beta.weight, 0.0)
            # nn.init.constant_(block.adaptive_norm3.beta.bias, 0.0)

            nn.init.constant_(block.mlp[0].weight, 0)
            nn.init.constant_(block.mlp[0].bias, 0)

        # Init projection out
        nn.init.xavier_uniform_(self.out_proj[0].weight)

    def partially_init_from_pretrained(self, ckpt):
        pretrained_model = ckpt['model_state_dict']
        print("Copy only transformer weights from pretrained model")
        for source_parameter, target_parameter in zip(pretrained_model.keys(), self.state_dict().keys()):
            if source_parameter == target_parameter and \
                    self.state_dict()[target_parameter].size() == pretrained_model[source_parameter].size()\
                    and "transformer" in source_parameter:
                print("copying:", source_parameter, self.state_dict()[target_parameter].size())
                self.state_dict()[target_parameter].data.copy_(pretrained_model[source_parameter])

    def forward(self, x, y, drop_label=None):
        b, seq_len = x.size()
        w, h = self.input_size, self.input_size

        # Drop the text-label y --> b, seq_len, cond_dim
        # if drop_label is not None:
        #     drop_label = drop_label.to(y.device)
        #     y = torch.where(drop_label.view(-1, 1, 1), torch.zeros_like(y), y)
        # drop_label逻辑改写
        if drop_label is not None:
            drop_label = drop_label.to(y.device)
            y = torch.where(drop_label.view(-1, 1, 1), torch.zeros_like(y), y)
            # 如果整个batch都是drop，则cond=None以避免NaN
            if drop_label.all():
                y = None
        # 降维 text token
        #y = self.text_proj(y)  # 将 y 的维度从 (batch_size, seq_len, 1024) 转换为 (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(x).any(), "Input x contains NaN values"
        # assert not torch.isinf(x).any(), "Input x contains Inf values"
        # if y is not None:
        #     assert not torch.isnan(y).any(), "Input y contains NaN values"
        #     assert not torch.isinf(y).any(), "Input y contains Inf values"
        # position embedding
        pos = torch.arange(0, w * h, dtype=torch.long, device=x.device)  # shape (t)
        pos = self.pos_emb(pos)

        # 图像 token 的嵌入
        x = self.tok_emb(x) + pos  # b, w*h, c
        x = self.first_norm(x)

        # reshape, proj to smaller space, reshape (patchify!)
        x = x.transpose(1, 2).contiguous().view(b, self.c, w, h)  # b, c, w, h
        x = self.in_proj(x)  # b, hidden, w // proj, h // proj
        x = x.view(b, self.hidden_dim, -1).transpose(1, 2).contiguous()  # b, (w // proj * h // proj), hidden

        # # 生成 Attention Mask
        # attention_mask = (y.abs().sum(dim=-1) > 0).float()  # [batch_size, seq_len]
        # attention mask安全检查
        if y is not None:
            attention_mask = (y.abs().sum(dim=-1) > 0).float()
            if attention_mask.sum() == 0:
                attention_mask = None
        else:
            attention_mask = None
        if attention_mask is not None:
            assert not torch.isnan(attention_mask).any(), "Attention mask contains NaN values"
            assert not torch.isinf(attention_mask).any(), "Attention mask contains Inf values"
            assert attention_mask.sum() > 0, "Attention mask is all zeros"
        # print(f"Forward called with x shape: {x.shape}, y shape: {y.shape}, attention_mask shape: {attention_mask.shape}")
        # Transformer Encoder
        x = self.transformer(x, y, attention_mask=attention_mask)  # b, (w // proj * h // proj), hidden

        x = x.transpose(1, 2).contiguous().view(b, self.hidden_dim, w // self.proj, h // self.proj)  # b, hidden, w // proj, h // proj
        x = self.out_proj(x)  # b, hidden//proj**2, w, h
        x = x.view(b, self.c, -1).transpose(1, 2).contiguous()  # b, w * h, hidden//proj**2

        x = self.last_norm(x)  # normalize before final prediction

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias
        # print({
        #     "x_mean": x.mean().item(),
        #     "x_std": x.std().item(),
        #     "logit_nan": torch.isnan(logit).any().item()
        # })

        return logit


if __name__ == "__main__":
    #  hidden_dim 是 Transformer 中的主要隐藏层维度，表示每个 token 的嵌入向量的维度
    #  mlp_dim（前馈网络的隐藏层中间维度）更高的非线性表达能力，使模型能够捕获更复杂的特征。
    model = Transformer(input_size=32, c=1024, hidden_dim=1024, cond_dim=2048,
                        codebook_size=16384, depth=24, heads=16, mlp_dim=1024*4,
                        dropout=0., proj=2)

    code = torch.randint(0, 16384, size=(2, 32, 32))
    txt = torch.randn(2, 120, 2048)
    drop_label = (torch.rand(2) < 0.1)

    c = model(x=code, y=txt, drop_label=drop_label)
    print(c.size())

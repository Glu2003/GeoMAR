import os, math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import pytorch_lightning as pl
from main_GeoMAR import instantiate_from_config
from alignment_hq import alignmodel
from ..modules.vqvae.utils import get_roi_regions
import pyiqa
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim.lr_scheduler import LambdaLR

class MaskedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mask_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_token_id = mask_token_id or vocab_size
        
        # 扩展词汇表包含掩码token
        extended_vocab_size = vocab_size + 1
        self.word_embeddings = nn.Embedding(extended_vocab_size, embedding_dim)
        
        # 使用截断正态分布初始化掩码token嵌入
        with torch.no_grad():
            self.word_embeddings.weight[self.mask_token_id].normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        # 确保索引在有效范围内
        valid_ids = torch.clamp(input_ids, 0, self.mask_token_id)
        return self.word_embeddings(valid_ids)
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
class TransformerSALayer(nn.Module): 
    # 实现自注意力机制和前馈神经网络mlp的Transformer编码器层
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout) # 多头注意力机制
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        # LayerNormapplied to the sums of the self attention and the input embedding
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        #激活函数
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) #将位置编码加到输入query和key上
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) 

        # ffn 
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2) #残差连接
        return tgt


class GeoMARModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path_HQ=None, # HQ checkpoint path
                 ckpt_path_LQ=None, # LQ checkpoint path
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000],

                 mask_token_id=1024,  # 新增：掩码标记ID
                 timesteps=8,        # 新增：推理时的迭代步数
                 mask_scheduling_method='cosine',  # 掩码调度方法
                 use_alignment=True,                           # 添加是否使用对齐的标志
                 is_training=True                      
                 ):
        super().__init__()

        # import pdb
        # pdb.set_trace()
        # self.scc_weight=0.001
        self.mask_token_id = mask_token_id
        self.timesteps = timesteps
        # self.num_iter = 8 # 训练时使用的迭代步数，可以小于推理时的步数
        self.mask_scheduling_method = mask_scheduling_method
        self.vocab_size = 1024  # 词汇表大小，与codebook_size相同
        print("timesteps = ", self.timesteps)    
        # 掩码调度函数
        # self.mask_schedule = lambda t, method='cosine': self._get_mask_ratio(t, method)
        
        # 新增：掩码调度函数
        # self.mask_schedule = lambda t: cosine_schedule(t)

        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig)

        lossconfig['params']['distill_param'] = ddconfig['params']
        # get the weights from HQ and LQ checkpoints
        if (ckpt_path_HQ is not None) and (ckpt_path_LQ is not None):
            print('loading HQ and LQ checkpoints')
            self.init_from_ckpt_two(
                ckpt_path_HQ, ckpt_path_LQ, ignore_keys=ignore_keys)

        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']

        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step


        # self.cross_attention = MultiHeadAttnBlock(in_channels=256,head_size=8)

        # codeformer code-----------------------------------
        dim_embd=512
        n_head=8
        n_layers=9 # transformer层数
        codebook_size=1024
        latent_size=256
        concat_size=512  # 连接后的序列长度

        connect_list=['32', '64', '128', '256']
        fix_modules=['quantize','generator']
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2
        self.feat_emb = nn.Linear(256, self.dim_embd)
        self.position_emb = nn.Parameter(torch.zeros(512, self.dim_embd))
       
        # 使用专用掩码嵌入层替代普通嵌入层
        self.token_emb = MaskedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.dim_embd,
            mask_token_id=self.mask_token_id
        )
        # transformer ft_layers由n_layers个TransformerSALayer组成
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head 用于预测索引的线性层
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        self.use_alignment = use_alignment
        # self.text_folder = text_folder
        
        # 如果使用对齐，初始化对齐模型
        self.alignmodel = alignmodel(
                dim=256,
                is_training=is_training,
            )

         # ==================== 初始化所有验证指标 ====================
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)

       
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_from_ckpt_two(self, path_HQ, path_LQ, ignore_keys=list()):
        """使用直接复制方式从HQ和LQ检查点加载权重到各个组件"""
        print('='*50)
        print('加载HQ和LQ检查点...')
        
        try:
            # 加载HQ检查点
            print(f"加载HQ检查点: {path_HQ}")
            sd_HQ = torch.load(path_HQ, map_location="cpu")
            if "state_dict" in sd_HQ:
                sd_HQ = sd_HQ["state_dict"]
            
            # 加载LQ检查点
            print(f"加载LQ检查点: {path_LQ}")
            sd_LQ = torch.load(path_LQ, map_location="cpu")
            if "state_dict" in sd_LQ:
                sd_LQ = sd_LQ["state_dict"]
            
            # 打印原始键数量
            print(f"HQ检查点键数量: {len(sd_HQ)}")
            print(f"LQ检查点键数量: {len(sd_LQ)}")
            
            # ----- 第1部分：定义组件映射 -----
            # 键格式: (检查点中的前缀, 模型中的组件, 检查点对象)
            component_mapping = [
                # HQ组件 - 从HQ检查点加载
                ('vqvae.quantize', self.vqvae.HQ_quantize, sd_HQ),
                ('vqvae.encoder', self.vqvae.HQ_encoder, sd_HQ),
                ('vqvae.quant_conv', self.vqvae.HQ_quant_conv, sd_HQ),
                ('vqvae.decoder', self.vqvae.decoder, sd_HQ),
                ('vqvae.post_quant_conv', self.vqvae.post_quant_conv, sd_HQ),
                
                # LQ组件 - 从LQ检查点加载
                ('vqvae.encoder', self.vqvae.encoder, sd_LQ),
                ('vqvae.quant_conv', self.vqvae.quant_conv, sd_LQ),
            ]
            
            # ----- 第2部分：加载各组件关键权重 -----
            # 直接加载embedding - 这是最关键的部分
            hq_embed_key = 'vqvae.quantize.embedding.weight'
            if hq_embed_key in sd_HQ:
                print(f"\n直接加载HQ codebook embedding权重:")
                print(f"  检查点中embedding: min={sd_HQ[hq_embed_key].min().item():.4f}, max={sd_HQ[hq_embed_key].max().item():.4f}")
                self.vqvae.HQ_quantize.embedding.weight.data.copy_(sd_HQ[hq_embed_key])
                print(f"  加载后embedding: min={self.vqvae.HQ_quantize.embedding.weight.min().item():.4f}, max={self.vqvae.HQ_quantize.embedding.weight.max().item():.4f}")
            else:
                print(f"警告: 未找到HQ embedding权重键 '{hq_embed_key}'")
            
            # ----- 第3部分：逐组件加载其余权重 -----
            loaded_params = 0
            
            for prefix, component, checkpoint in component_mapping:
                component_loaded = 0
                
                # 对于每个组件遍历其所有命名参数
                for name, param in component.named_parameters():
                    # 构建检查点中的对应键名
                    ckpt_key = f"{prefix}.{name}"
                    
                    # 检查键是否存在于检查点中
                    if ckpt_key in checkpoint:
                        # 直接复制权重
                        param.data.copy_(checkpoint[ckpt_key])
                        component_loaded += 1
                        loaded_params += 1
                
                # 打印每个组件加载的参数数量
                total_params = sum(1 for _ in component.parameters())
                # print(f"组件 {prefix}: 加载参数 {component_loaded}/{total_params}")
                print(f"组件 {prefix} ({'HQ' if checkpoint is sd_HQ else 'LQ'}): 加载参数 {component_loaded}/{total_params}")

            # ----- 第4部分: 验证权重加载 -----

            
            # ----- 第5部分: 打印最终HQ量化器统计信息 -----
            print('加载检查点完成')
            print('='*50)
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            import traceback
            traceback.print_exc()
            raise

    # 3. 添加掩码调度函数
    def _get_mask_ratio(self, t, method='cosine'):
        """获取不同时间步的掩码比例"""
        if method == 'cosine':
            return math.cos(math.pi/2 * t)
        elif method == 'linear':
            return 1.0 - t
        else:
            return math.cos(math.pi/2 * t)  # 默认使用余弦调度  

        # ============ 掩码自回归部分 ============
    
    def maskgit_train_step(self, lq_feat, gt_indices):
        """
        lq_feat: [B, L, C]
        gt_indices: [B, L]  长度 L=256
        """
        device = lq_feat.device
        B, L = gt_indices.shape
         # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,B,1)
        
        feat_emb = self.feat_emb(lq_feat.permute(1,0,2)) # [seq_len, batch, embed_dim]
        
        # --------------------------------------------------- 
        # 1. 随机 mask ratio r ∈ [0,1] 
        # --------------------------------------------------- 
        r = torch.rand(1, device=device).item()
        mask_ratio = self._get_mask_ratio(r, self.mask_scheduling_method)
            # 精确选择掩码数量，而非随机概率
        num_mask = max(1, int(mask_ratio * L))
        # ---------------------------------------------------
        # 优化后的 Mask 采样策略
        # ---------------------------------------------------
        # 1. 权重调整：更重视结构生成 (step 0, 1)
        # weights = torch.tensor([3, 2, 2, 2, 2, 2, 1, 1], device=device, dtype=torch.float)
        # step_id = torch.multinomial(weights, 1).item()
        # num_steps = 8

        # # 2. 连续区间采样：覆盖 [0, 1] 全范围
        # t_min = step_id / num_steps
        # t_max = (step_id + 1) / num_steps
        # t = torch.empty(1, device=device).uniform_(t_min, t_max).item()

        # # 3. 显式全 Mask 增强 (10% 概率)
        # if torch.rand(1) < 0.1: 
        #     mask_ratio = 1.0
        # else:
        #     mask_ratio = self._get_mask_ratio(t) # Cosine schedule
    
        # # 4. 计算 mask 数量 (含概率取整)
        # num_mask_float = mask_ratio * L
        # num_mask = int(num_mask_float)
        # if torch.rand(1) < (num_mask_float - num_mask):
        #     num_mask += 1
            
        # num_mask = max(1, min(L, num_mask)) # 确保至少 mask 1 个，至多 L 个
        # # ---------------------------------------------------
        # # 1. 随机 mask ratio r ∈ [0,1]
        # # ---------------------------------------------------
        # # r = torch.rand(1, device=device).item()
        # # inference-aligned t
        # t_centers = torch.tensor(
        #     [0.0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8],
        #     device=device
        # )

        # weights = torch.tensor(
        #     [1, 1, 2, 3, 3, 2, 1, 1],
        #     device=device,
        #     dtype=torch.float
        # )

        # step_id = torch.multinomial(weights, 1)
        # # t_center = t_centers[step_id]

        # t_center = t_centers[step_id].item()  # Tensor → float

        # delta = 1.0 / 16

        # t = torch.empty(1, device=device).uniform_(
        #     max(0.0, t_center - delta),
        #     min(7/8, t_center + delta)
        # ).item()


        # mask_ratio = self._get_mask_ratio(t)
        # # print(f"训练步掩码比例: {mask_ratio:.4f} (t={t:.4f})")
        # num_mask = max(1, int(mask_ratio * L))


        # ---------------------------------------------------
        # 2. 随机采样 mask 的位置
        # ---------------------------------------------------
        rand_perm = torch.rand(B, L, device=device).argsort(dim=-1)
        mask_pos = rand_perm[:, :num_mask]             # [B, num_mask]
        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, mask_pos, True)

        # ---------------------------------------------------
        # 3. 生成 masked token 序列
        # ---------------------------------------------------
        input_tokens = gt_indices.clone()
        input_tokens[mask_bool] = self.mask_token_id   # 被 mask 的位置替换为 mask token
        # ---------------------------------------------------
        # 4. token embedding + lq_feat embedding
        # ---------------------------------------------------
        # ---- LQ 特征 embedding （作为 prefix）
        # ---- token embedding
        tok_emb = self.token_emb(input_tokens).permute(1, 0, 2) # [B,seq_len,dim] -> [seq_len,B,dim]
        # ---- concat prefix + token
        input_emb = torch.cat([feat_emb, tok_emb], dim=0) # [2*seq_len, batch, embed_dim]

        # ---------------------------------------------------
        # 5. Transformer forward（全并行）
        # ---------------------------------------------------
        # Transformer前向传播
        query_emb = input_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)  # 直接使用完整的位置编码
        
        # 只取HQ令牌部分的输出进行预测
        hq_output = query_emb[L:]  # 只取后半部分 [seq_len, batch, embed_dim]

        # ---------------------------------------------------
        # 6. 预测 logits
        # ---------------------------------------------------
        logits = self.idx_pred_layer(hq_output)       # [seq_len, B, vocab_size]
        logits = logits.permute(1, 0, 2)  # [B, seq_len, vocab_size]
        # ---------------------------------------------------
        # 7. CE loss with mask positions only
        # ---------------------------------------------------
        loss = F.cross_entropy(
           logits[mask_bool], gt_indices[mask_bool],label_smoothing=0.1)

        return loss, logits

    @torch.no_grad()
    def maskgit_generate(self, x,  filenames, gt=None):
        """
        推理：iterative parallel decoding.
        输入：LQ feature
        输出：最终生成的 token 序列 [B, L]
        """
        
        gt_indices = None
        quant_gt = None
        L2_loss = torch.tensor(0.0, device=x.device)
        if gt is not None:   #检查是否提供gt
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt) #将gt编码
        
        # LQ feature from LQ encoder and quantizer 
        z_hs = self.vqvae.encoder(x)
        
        z_h = self.vqvae.quant_conv(z_hs['out'])
       
        # origin HQ codebook for index 用原始HQcodebook进行索引
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.HQ_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = indices

        # 添加特征对齐处理 =============================================
        # 使用特征对齐
        batch_size, c, h, w = z_h.shape
        z_seq = z_h.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        
        if self.alignmodel is not None and filenames is not None:
            # 类型检查 - 避免处理Tensor类型的文件路径
            if isinstance(filenames, torch.Tensor):
                print("检测到Tensor类型的文件路径，跳过特征对齐")
                lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            else:
                # 正常处理字符串路径
                batch_with_gt_path = {"gt_path": filenames}
                text_features = self.alignmodel.get_text_features(z_h, batch=batch_with_gt_path)
               
                # 检测并匹配数据类型
                model_dtype = next(self.alignmodel.parameters()).dtype
                z_seq = z_seq.to(dtype=model_dtype)
                text_features = text_features.to(dtype=model_dtype)
        
                
                lq_feat,_ = self.alignmodel(z_seq, text_features)

                # 转回原始数据类型以保持一致性
                lq_feat = lq_feat.to(dtype=z_h.dtype)
            
        else:
            # 如果没有对齐模型，直接使用原始特征
            lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            print("未使用对齐模型，直接使用原始特征")

        B, L, C = lq_feat.shape
        device = lq_feat.device
        # 初始化：全 mask
        current_hq_tokens = torch.full((B, L), fill_value=self.mask_token_id, dtype=torch.long, device=device)
        # print("当前HQ tokens:", current_hq_tokens)
        # prefix LQ 特征 embedding（与训练一致）
        feat_emb = self.feat_emb(lq_feat.permute(1,0,2)) # [seq_len, batch, embed_dim]
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, B, 1)  # [seq_len, B, dim]

        # # ------------------------------
        # 多步 iterative reveal
        # ------------------------------
        total_ce = 0.0
 
        for t in range(self.timesteps):
            # print("推理步骤 {}/{}".format(t+1, self.timesteps))
            
            # token embedding
            tok_emb = self.token_emb(current_hq_tokens).permute(1, 0, 2)  # [B,seq_len,dim] -> [seq_len,B,dim]    
            query_emb = torch.cat([feat_emb, tok_emb], dim=0) # [2*seq_len, batch, embed_dim]
            
            
            # Transformer前向传播
            for layer in self.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)

            hq_output = query_emb[L:]  # 只取后半部分 [seq_len, batch, embed_dim]

            logits = self.idx_pred_layer(hq_output)
            logits = logits.permute(1, 0, 2)

            # probs = F.softmax(logits, dim=-1)
            # if t ==7:
            k_max = 1024  # 早期步骤使用的最大 k 值，增加多样性
            k_min = 1 # 后期步骤使用的最小 k 值 (k=1 等同于 argmax)，确保一致性
            if t<2:
                current_k=k_max
            # --- 2. 根据当前步骤 t 计算 current_k (线性衰减) ---
            # progress_ratio 从 0.0 (t=0) 变化到 1.0 (t=timesteps-1)
            else:

                progress_ratio = (t + 1) / self.timesteps  # 从 0.0 到 1.0
                # cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_ratio)) 
                # current_k = k_min + (k_max - k_min) * cosine_decay
                # current_k = int(round(current_k))
                # current_k = max(k_min, current_k)
                current_k = k_max - (k_max - k_min) * progress_ratio
                current_k = int(current_k)
                current_k = max(k_min, current_k) # 确保 k 至少为 1
            # pred_tokens = torch.argmax(probs, dim=-1)        # [B, L]
            # confidence = torch.max(probs, dim=-1).values  # [B, L]
 
            topk_logits, topk_indices = torch.topk(logits, k=current_k, dim=-1)

            # 步骤 2: 对 top-k logits 应用 softmax，创建新的概率分布
            # topk_probs: [B, L, k]
            topk_probs = F.softmax(topk_logits, dim=-1)

            # 步骤 3: 从这个新的 k 维分布中随机采样
            # topk_probs.shape: [B, L, k]
            # torch.multinomial 需要 2D 输入 [N, C]，所以我们先 reshape
            B, L, _ = topk_probs.shape
            # topk_sample_idx: [B * L, 1]，每个元素的值在 [0, k-1] 之间
            topk_sample_idx = torch.multinomial(topk_probs.view(B * L, -1), num_samples=1)

            # 将采样得到的索引 reshape 回原来的 [B, L] 形状
            # topk_sample_idx: [B, L]
            topk_sample_idx = topk_sample_idx.view(B, L)

            # 步骤 4: 使用采样到的索引从 topk_indices 中选出最终的 token
            # topk_indices: [B, L, k], topk_sample_idx: [B, L] -> [B, L, 1]
            # pred_tokens: [B, L]
            pred_tokens = torch.gather(topk_indices, dim=-1, index=topk_sample_idx.unsqueeze(-1)).squeeze(-1)

            # 步骤 5: 计算置信度
            # 关键：置信度必须是最终选中的 token 在 *原始* 概率分布中的概率
            #
            # 首先，获取原始的完整概率分布
            probs = F.softmax(logits, dim=-1)
            # 然后，使用最终选出的 pred_tokens 作为索引，在 probs 中查找它们的概率
            # pred_tokens: [B, L] -> [B, L, 1]
            # confidence: [B, L]
            confidence = torch.gather(probs, dim=-1, index=pred_tokens.unsqueeze(-1)).squeeze(-1)


            # ---- freeze revealed tokens ----
            mask = (current_hq_tokens == self.mask_token_id)
            # print("当前掩码位置 mask:", mask)
            confidence = confidence.masked_fill(~mask, -float("inf"))
            

            # 计算本次应 reveal 的比率
            progress_ratio = (t + 1) / self.timesteps
            if t == self.timesteps -1:
                mask_ratio_t_1 = 0.0  # 最后一步揭示所有token
            else:   
                mask_ratio_t_1 =self._get_mask_ratio(progress_ratio, self.mask_scheduling_method)
            # print("mask_ratio_t_1:", mask_ratio_t_1)
            
            batch_step_ce = 0.0
            # 全局处理置信度
            batch_tokens = []
            # temperature = 2 * (1.0 - t/self.timesteps)**2  # 温度参数
            # if gt_indices is not None:
            #     # 只在 mask 位置上评估
            #     gt_masked = gt_indices[mask]        # [N_mask]
            #     pred_masked = pred_tokens[mask]     # [N_mask]

            #     if gt_masked.numel() > 0:
            #         step_error = (pred_masked != gt_masked).float().sum()
            #         step_count = gt_masked.numel()
            #     else:
            #         step_error = torch.tensor(0.0, device=device)
            #         step_count = 0

            #     step_error_list.append(step_error)
            #     step_count_list.append(step_count)
                # print(f"步骤 {t+1}/{self.timesteps} - step_error: {step_error.item()}, step_count: {step_count}")

            for b in range(B):
                cur_tokens = current_hq_tokens[b]
                # print("当前HQ tokens:", cur_tokens)
                current_mask = mask[b]
                # print("当前掩码位置:", current_mask)
                unmasked_count = (~current_mask).sum().item()
                # print("已揭示token数量:", unmasked_count)
                num_reveal = int((1 - mask_ratio_t_1) * L)
                # print("下一步应揭示token数量:", num_reveal)

                # 计算还需要揭示多少token
                tokens_to_reveal = max(0, num_reveal - unmasked_count)
                # print("本步需揭示token数量:", tokens_to_reveal)
                # 创建新token状态，首先复制所有已揭示的token
                new_tokens =  cur_tokens.clone()
                
                 # 如果需要揭示更多token且还有掩码位置
                if tokens_to_reveal > 0 and current_mask.any():
                    # 只考虑掩码位置的置信度
                    mask_positions = torch.where(current_mask)[0]
                    masked_confidence = confidence[b][mask_positions]


                    # 选择置信度最高的k个位置进行揭示
                    _, top_indices = torch.topk(masked_confidence, k=min(tokens_to_reveal, len(mask_positions)))
                    positions_to_reveal = mask_positions[top_indices]
                                
                    # 揭示选中的位置
                    new_tokens[positions_to_reveal] = pred_tokens[b][positions_to_reveal]
                    
                    if gt_indices is not None:
                        # 计算本步 CE
                        step_ce = F.cross_entropy(logits[b, positions_to_reveal], gt_indices[b, positions_to_reveal])
                        batch_step_ce += step_ce            
                batch_tokens.append(new_tokens)
                
            # 更新当前tokens
            current_hq_tokens = torch.stack(batch_tokens)
            # print("更新后的HQ tokens:", current_hq_tokens)
            total_ce += batch_step_ce / B
            # ===== 4. 最终安全检查 =====
        # 确保没有掩码标记
        mask_positions = (current_hq_tokens == self.mask_token_id)
        if mask_positions.any():
            print(f"警告: 仍有{mask_positions.sum().item()}个未揭示的掩码位置，进行随机填充")
            current_hq_tokens[mask_positions] = torch.randint(
                0, self.vocab_size, (mask_positions.sum().item(),), device=device
            )
        
        # 确保索引范围有效
        current_hq_tokens = torch.clamp(current_hq_tokens, 0, self.vocab_size-1)
         # 4. decode 成图像
        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(current_hq_tokens.view(-1), 
                                                           shape=[B, 16,16,256])
       
        dec_img = self.vqvae.decode(quant_feat)
        lq_feat = lq_feat.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        if quant_gt is not None:
            L2_loss = F.mse_loss(lq_feat, quant_gt)

        return current_hq_tokens, dec_img, total_ce, L2_loss 


   

                    # 生成 Gumbel 噪声
                    # u = torch.rand_like(masked_confidence)
                    # gumbel_noise = -torch.log(-torch.log(u + 1e-7) + 1e-7)

                    # 线性衰减噪声并加到 log(confidence)
                    # noisy_masked_confidence = torch.log(masked_confidence + 1e-9) + r_temp * (1 - ratio) * gumbel_noise
                    # noisy_masked_confidence = torch.log(masked_confidence + 1e-9) + temperature * gumbel_noise
    
        
    def forward(self, input, gt=None,filenames=None, save_features=False, features_dir="features"):
        
        if gt is not None:   #检查是否提供gt
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt) #将gt编码
        
        # LQ feature from LQ encoder and quantizer 
        z_hs = self.vqvae.encoder(input)
        
        z_h = self.vqvae.quant_conv(z_hs['out'])
       
        # origin HQ codebook for index 用原始HQcodebook进行索引
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.HQ_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = indices

        if gt is None:
            quant_gt = quant_z
            gt_indices = z_indices
            self.alignmodel.eval()
            self.alignmodel.set_eval_mode(True)

        # 添加特征对齐处理 =============================================
        # 使用特征对齐
        batch_size, c, h, w = z_h.shape
        z_seq = z_h.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        
        if self.alignmodel is not None and filenames is not None:
            # 类型检查 - 避免处理Tensor类型的文件路径
            if isinstance(filenames, torch.Tensor):
                print("检测到Tensor类型的文件路径，跳过特征对齐")
                lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            else:
                # 正常处理字符串路径
                batch_with_gt_path = {"gt_path": filenames}
                text_features = self.alignmodel.get_text_features(z_h, batch=batch_with_gt_path)
               
                # 检测并匹配数据类型
                model_dtype = next(self.alignmodel.parameters()).dtype
                z_seq = z_seq.to(dtype=model_dtype)
                text_features = text_features.to(dtype=model_dtype)
        
                # [MODIFIED] 调用 forward 并请求对比特征
                lq_feat, contrastive_loss  = self.alignmodel(z_seq, text_features)

                # 转回原始数据类型以保持一致性
                lq_feat= lq_feat.to(dtype=z_h.dtype)
            
        else:
            # 如果没有对齐模型，直接使用原始特征
            lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            print("未使用对齐模型，直接使用原始特征")


       
    #     # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
    #     pos_emb = self.position_emb.unsqueeze(1).repeat(1,z_h.shape[0],1)
       
    #     # BCHW -> BC(HW) -> (HW)BC
    #     feat_emb = self.feat_emb(lq_feat.permute(1,0,2))
       
    #     query_emb = feat_emb
    #     # Transformer encoder
    #     for layer in self.ft_layers:
    #         query_emb = layer(query_emb, query_pos=pos_emb)
    #    # output logits
    #     logits = self.idx_pred_layer(query_emb) # (hw)bn
    #     logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

    #     BCE_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gt_indices.view(-1))

    #     # logits: [B, HW, N]
    #     probs = F.softmax(logits, dim=-1)

    #     k = 1 # 你想用的 top-k
    #     topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    #     # topk_probs:  [B, HW, k]
    #     # topk_indices: [B, HW, k]

    #     # 在 top-k 内重新归一化
    #     topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    #     # multinomial 需要 2D：[N, C]
    #     B, HW, _ = topk_probs.shape
    #     sampled_idx = torch.multinomial(
    #         topk_probs.reshape(-1, k),  # [(B*HW), k]
    #         num_samples=1
    #     )  # [(B*HW), 1]

    #     # 映射回原始 codebook index
    #     top_idx = topk_indices.reshape(-1, k).gather(
    #         dim=1,
    #         index=sampled_idx
    #     ).reshape(B, HW)  # [B, HW]
    #     # get index from origin HQ codebook
    #     quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(top_idx.reshape(-1), shape=[z_h.shape[0],16,16,256])
        
    #     lq_feat = lq_feat.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
    #     L2_loss = F.mse_loss(lq_feat, quant_gt)
        
    #     # preserve gradients
    #     quant_feat = lq_feat + (quant_feat - lq_feat).detach()
    #     dec = self.vqvae.decode(quant_feat)

         # ============ 使用迭代式掩码自回归替代原方法 ============
        BCE_loss, logits = self.maskgit_train_step(
            lq_feat=lq_feat, 
            gt_indices=gt_indices
        )

        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)


        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(top_idx.reshape(-1), shape=[z_h.shape[0],16,16,256])
        
   
        lq_feat = lq_feat.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        L2_loss = F.mse_loss(lq_feat, quant_gt)
        
        # 保留梯度 - 使用形状匹配的张量
        quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        dec = self.vqvae.decode(quant_feat)
        # 获取特征并解码
        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(
            top_idx.reshape(-1), 
            shape=[z_h.shape[0], 16, 16, 256]
        )

        # # 将quant_feat转换为与lq_feat相同的形状
        # batch_size, h, w, c = z_h.shape[0], 16, 16, 256
        # quant_feat_reshaped = quant_feat.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)

        # # 将quant_gt转换为与lq_feat相同的形状以计算L2损失
        # batch_size_gt, c_gt, h_gt, w_gt = quant_gt.shape
        # quant_gt_reshaped = quant_gt.permute(0, 2, 3, 1).reshape(batch_size_gt, h_gt*w_gt, c_gt)

        # # 保留梯度 - 使用形状匹配的张量
        # quant_feat_reshaped = z_seq + (quant_feat_reshaped - z_seq).detach()

        # # 计算 L2 损失 - lq_feat 和 quant_gt_reshaped 现在有相同的形状
        # L2_loss = F.mse_loss(z_seq, quant_gt_reshaped)
        # # 将形状转回用于解码器
        # quant_feat = quant_feat_reshaped.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        dec = self.vqvae.decode(quant_feat)

        # contrastive_loss = torch.tensor(0.0, device=input.device)

        return dec, BCE_loss, L2_loss, z_info, z_hs, z_h, quant_gt, z_dictionary, contrastive_loss
    




    @torch.no_grad()
    def encode_to_gt(self, gt):
        quant_gt, _, info, hs, h, dictionary = self.vqvae.HQ_encode(gt)
        indices = info[2].view(quant_gt.shape[0], -1)
        return quant_gt, indices, info, hs, h, dictionary

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.alignmodel.train()
        self.alignmodel.set_eval_mode(False)
        if optimizer_idx == None:
            optimizer_idx = 0

        x = batch[self.image_key]
        gt = batch['gt']
        filenames = batch.get('gt_path', None)
        xrec, BCE_loss, L2_loss, info, hs,_,_,_, contrastive_loss = self(x, gt, filenames)

        # qloss = BCE_loss + 10*L2_loss + contrastive_loss

        if self.image_key != 'gt':
            x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(
                x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        if optimizer_idx == 0:

            aeloss = BCE_loss + 10*L2_loss + contrastive_loss

            rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous()))

            log_dict_ae = {
                    "train/total_aeloss": aeloss.detach().mean(),
                   "train/BCE_loss": BCE_loss.detach().mean(),
                   "train/L2_loss": L2_loss.detach().mean(),
                   "train/Rec_loss": rec_loss.detach().mean(),
                   "train/Contrastive_loss": contrastive_loss.detach().mean()
                }
            
            # bce_loss = log_dict_ae["train/BCE_loss"]
            # self.log("BCE_loss", bce_loss, prog_bar=True,
            #          logger=True, on_step=True, on_epoch=True)
            
            # l2_loss = log_dict_ae["train/L2_loss"]
            # self.log("L2_loss", l2_loss, prog_bar=True,
            #          logger=True, on_step=True, on_epoch=True)
            
            # Rec_loss = log_dict_ae["train/Rec_loss"]
            # self.log("Rec_loss", Rec_loss, prog_bar=True,
            #          logger=True, on_step=True, on_epoch=True)

            # contrastive_loss_log = log_dict_ae["train/Contrastive_loss"]
            # self.log("Contrastive_loss", contrastive_loss_log, prog_bar=True,
            #          logger=True, on_step=True, on_epoch=True)


            self.log_dict(
            log_dict_ae,
            prog_bar=True,    # 主要指标显示在进度条
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True    # 分布式同步
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.disc_start <= self.global_step:

            # left eye
            if optimizer_idx == 2:
                # discriminator
                disc_left_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                          last_layer=None, split="train")
                self.log("train/disc_left_loss", disc_left_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_left_loss

            # right eye
            if optimizer_idx == 3:
                # discriminator
                disc_right_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_right_loss", disc_right_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_right_loss

            # mouth
            if optimizer_idx == 4:
                # discriminator
                disc_mouth_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_mouth_loss", disc_mouth_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_mouth_loss

    def validation_step(self, batch, batch_idx):
       
        # self.eval()  # 切换 eval
       
        x = batch[self.image_key]

        gt = batch['gt']
        filenames = batch.get('gt_path', batch.get('filename', None))

        xrec, BCE_loss, L2_loss, info, hs,_,_,_, contrastive_loss = self(x, gt, filenames)

        # qloss = BCE_loss + L2_loss

        if self.image_key != 'gt':
            x = batch['gt']


        xrec = torch.clamp(xrec, -1.0, 1.0) 
        xrec_norm = torch.clamp((xrec + 1.0) / 2.0, 0.0, 1.0)
        gt_norm = torch.clamp((gt + 1.0) / 2.0, 0.0, 1.0)

        self.fid_metric.update(gt_norm, real=True)
        self.fid_metric.update(xrec_norm, real=False)
        self.psnr_metric.update(xrec_norm, gt_norm)
        self.ssim_metric.update(xrec_norm, gt_norm)
       
         # ================= 计算 NIQE (无参考指标) =================
        # NIQE 可以按 batch 计算平均值
        try:
            if self.niqe_metric.device != xrec_norm.device:
                self.niqe_metric = self.niqe_metric.to(xrec_norm.device)
            
            # pyiqa 的 niqe 输入通常需要 [0, 1]
            val_niqe = self.niqe_metric(xrec_norm).mean()
        except Exception as e:
            if self.global_rank == 0:
                print(f"NIQE Error: {e}")
            val_niqe = torch.tensor(0.0, device=xrec_norm.device)
        # ===========================================================

        # try:
        #     if self.maniqa_metric.device != xrec_norm.device:
        #         self.maniqa_metric = self.maniqa_metric.to(xrec_norm.device)
            
        #     # pyiqa 的 niqe 输入通常需要 [0, 1]
        #     val_maniqa = self.maniqa_metric(xrec_norm).mean()
        # except Exception as e:
        #     if self.global_rank == 0:
        #         print(f"maniqa Error: {e}")
        #     val_maniqa = torch.tensor(0.0, device=xrec_norm.device)

        
        rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous())).mean()
        
        self.log("val_niqe", val_niqe.detach(), prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=False)
        # self.log("val_maniqa", val_maniqa.detach(), prog_bar=True,
        #          logger=True, on_step=False, on_epoch=True, sync_dist=False)    
                         
        log_dict_ae = {
                "val_BCE_loss": BCE_loss.detach().mean(),
                "val_L2_loss": L2_loss.detach().mean(),
                "val_Rec_loss": rec_loss.detach(),
            }

        # self.log_dict(log_dict_ae)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

        if self.global_rank == 0:
            # print(f'niqe: {val_niqe.item():.4f}')
            print(f'Validation Step {batch_idx}: niqe: {val_niqe.item():.4f}')

        return self.log_dict
    # def validation_step(self, batch, batch_idx):
    #     x = batch[self.image_key]
    #     gt = batch['gt']
    #     filenames = batch.get('gt_path', batch.get('filename', None))

    #     # 1. 生成图像并计算损失
    #     _, xrec, bce_loss, l2_loss = self.maskgit_generate(x=x, filenames=filenames, gt=gt)

    #     rec_loss = torch.abs(gt.contiguous() - xrec.contiguous()).mean()

    #     # 2. 图像归一化到 [0, 1] 范围，这是所有指标所期望的输入
    #     xrec_norm = torch.clamp((xrec + 1.0) / 2.0, 0.0, 1.0)
    #     gt_norm = torch.clamp((gt + 1.0) / 2.0, 0.0, 1.0)

    #     # 3. 更新 torchmetrics 指标的状态
    #     # FID 需要真实图像和生成图像来更新
    #     self.fid_metric.update(gt_norm, real=True)
    #     self.fid_metric.update(xrec_norm, real=False)
        
    #     # PSNR 和 SSIM 更新
    #     self.psnr_metric.update(xrec_norm, gt_norm)
    #     self.ssim_metric.update(xrec_norm, gt_norm)
        
    #     # 4. 记录每个 batch 的损失 (这部分保持不变)
    #     self.log_dict({
    #         "val_BCE_loss": bce_loss.detach(),
    #         "val_L2_loss": l2_loss.detach(),
    #         "val_Rec_loss": rec_loss.detach(),
    #     }, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    #     # 5. 收集生成的图像用于 pyiqa 指标计算
    #     # 只需收集 xrec_norm，因为 NIQE 和 MANIQA 是无参考指标
    #     self.val_outputs.append(xrec_norm)

    #     return self.log_dict

    def on_validation_epoch_end(self):
        # ===========================================================================
        # 1. 计算并记录 torchmetrics 指标 (这是最可能缺失的部分)
        # ===========================================================================
        try:
            val_fid = self.fid_metric.compute()
            val_psnr = self.psnr_metric.compute()
            val_ssim = self.ssim_metric.compute()

            # 重置指标状态，为下一个验证周期做准备
            self.fid_metric.reset()
            self.psnr_metric.reset()
            self.ssim_metric.reset()
        except Exception as e:
            # 如果计算出错，也给一个默认值，防止程序崩溃
            if self.global_rank == 0:
                print(f"Error computing torchmetrics: {e}")
            val_fid = torch.tensor(float('inf'), device=self.device) # 用一个很大的值
            val_psnr = torch.tensor(0.0, device=self.device)
            val_ssim = torch.tensor(0.0, device=self.device)

            
        self.trainer.callback_metrics["val_fid"] = val_fid
        self.trainer.callback_metrics["val_psnr"] = val_psnr
        self.trainer.callback_metrics["val_ssim"] = val_ssim


        if self.global_rank == 0:
            print(f"\nEpoch End Validation: FID={val_fid:.4f} PSNR={val_psnr:.4f} SSIM={val_ssim:.4f}\n")

    # def on_validation_epoch_end(self):
    #     # 1. 计算并记录 torchmetrics 指标
    #     val_fid = self.fid_metric.compute()
    #     val_psnr = self.psnr_metric.compute()
    #     val_ssim = self.ssim_metric.compute()

    #     # 重置指标状态，为下一个验证周期做准备
    #     self.fid_metric.reset()
    #     self.psnr_metric.reset()
    #     self.ssim_metric.reset()

    #     # 2. 计算并记录 pyiqa 指标
    #     # 将所有收集到的生成图像拼接成一个大的 batch
    #     all_xrec_norm = torch.cat(self.val_outputs, dim=0)
        
    #     try:
    #         val_niqe = self.niqe_metric(all_xrec_norm)
    #         val_maniqa = self.maniqa_metric(all_xrec_norm)
    #     except Exception as e:
    #         if self.global_rank == 0:
    #             print(f"Error computing NIQE/MANIQA: {e}")
    #         val_niqe = torch.tensor(float('nan'), device=self.device)
    #         val_maniqa = torch.tensor(float('nan'), device=self.device)
            
    #     # 清空收集的输出
    #     self.val_outputs.clear()

    #     # 3. 统一记录所有周期级别的指标
    #     log_dict_epoch = {
    #         "val_fid": val_fid,
    #         "val_psnr": val_psnr,
    #         "val_ssim": val_ssim,
    #         "val_niqe": val_niqe.mean(),  # .mean() 以防返回的是tensor
    #         "val_maniqa": val_maniqa.mean()
    #     }
        
    #     self.log_dict(log_dict_epoch, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    #     if self.global_rank == 0:
    #         print(f"\nEpoch End Validation: FID={val_fid:.4f}, NIQE={val_niqe.mean():.4f}, MANIQA={val_maniqa.mean():.4f}, PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}\n")
    
    def on_validation_epoch_start(self):
        self.alignmodel.set_eval_mode(True)
        

    def configure_optimizers(self):
        lr = self.learning_rate
        print(f"配置优化器，基础学习率: {lr}")
        normal_params = []
        special_params = []
        fixed_params = []
        fixed_parameter = 0
        test_count = 0
        # schedules = []
        # autoencoder part -------------------------------
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue

            if 'HQ' in name:
                special_params.append(param)
                fixed_parameter = fixed_parameter + 1
                continue
            if 'decoder' in name or 'post_quant_conv' in name or 'quantize' in name:
                test_count = test_count + 1
                # continue
                special_params.append(param)
                # print(name)
            else:
                normal_params.append(param)

        # 添加alignmodel参数 - 这是新增的部分
        if self.alignmodel is not None:
            print("将alignmodel参数添加到优化器中...")
            for name, param in self.alignmodel.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    normal_params.append(param)
                    print(f"添加alignmodel参数: {name}")
                
        # transformer part--------------------------------
        
        normal_params.append(self.position_emb)   
        
        for name, param in self.feat_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        for name, param in self.ft_layers.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 
        for name, param in self.token_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)

        for name, param in self.idx_pred_layer.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)                 
        
        # print('special_params', special_params)
        opt_ae_params = [{'params': normal_params, 'lr': lr}]

        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))

        optimizations = opt_ae

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]

            s2 = torch.optim.lr_scheduler.MultiStepLR(
                opt_l, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s3 = torch.optim.lr_scheduler.MultiStepLR(
                opt_r, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s4 = torch.optim.lr_scheduler.MultiStepLR(
                opt_m, milestones=self.schedule_step, gamma=0.1, verbose=True)
            schedules += [s2, s3, s4]
        # ----------------- Cosine Annealing LR -----------------
        # total_steps = 200_000  # 总迭代次数
        # scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt_ae,
        #     T_max=total_steps,
        #     eta_min=1e-5,
        #     verbose=True
        # )

        # ----------------- 返回 -----------------
        # 注意：这里不加 []，单个 optimizer + scheduler
        # return [opt_ae], [{'scheduler': scheduler_ae, 'interval': 'step'}]
        # return optimizations, schedules
        return optimizations

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, split, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
        # 检查是否有GT图像
        # has_gt = 'gt' in batch
        
        # if has_gt:
        gt = batch['gt'].to(self.device)
        filenames = batch.get('gt_path', batch.get('filename', None))
    
        if split == 'train':
            xrec, _, _, _, _, _, _, _,_ = self(x, gt, filenames)

        elif split == 'val':
            self.eval()  # 切换 eval
            self.alignmodel.set_eval_mode(True)
            xrec, BCE_loss, L2_loss, info, hs,_,_,_, contrastive_loss = self(x, gt, filenames)

        log["inputs"] = x
        log["reconstructions"] = xrec
        
        if self.image_key != 'gt':
            x = batch['gt']
            log["gt"] = x
        
        return log
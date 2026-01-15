# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.in_chans = in_chans
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print(f'encoder output shape: {x.shape}')
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        # print(f'decoder output shape: {x.shape}')
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*2]
        BVP_map = self.unpatchify(pred) # [N, C, H, W]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, BVP_map, mask

    def show_masked_img(self, imgs, mask):
        """
        imgs: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        N, C, H, W = imgs.shape
        L = H * W // self.patch_embed.patch_size[0] ** 2
        mask = mask.reshape(N, L, 1)
        imgs = imgs.reshape(N, C, H, W)
        imgs = imgs * (1 - mask)
        return imgs


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=8, num_heads=8,
        decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_masked_result_viridis(imgs, mask, model, num_samples=1):
        """
        可视化 Mask 结果：
        - 原图：viridis 配色
        - 掩码图：底色 viridis + 被掩码部分全白
        
        参数：
            imgs: 输入图像张量，形状 [N, C, H, W]（C=1 为灰度图，如 ST-map）
            mask: Mask 中间结果，形状 [N, L]（L=Patch总数）
            model: MAE 模型实例（用于获取 Patch 大小、图像尺寸）
            num_samples: 从批量中选择前 N 个样本绘制（默认 1）
        """
        # 1. 获取模型关键参数（Patch 大小、图像尺寸）
        patch_size = model.patch_embed.patch_size[0]  # 如 16（16x16 Patch）
        img_size = model.patch_embed.img_size[0]      # 如 224（输入图像边长）
        num_patches_per_side = img_size // patch_size # 每边 Patch 数量（如 224/16=14）

        # 遍历批量中的样本（默认绘制第一个样本）
        for idx in range(num_samples):
            # --------------------------
            # 步骤1：处理原始灰度图（归一化 + 适配 matplotlib）
            # --------------------------
            img_gray = imgs[idx].detach().cpu().numpy()  # 取单个样本：[1, 224, 224] → [224, 224]
            img_gray = img_gray.squeeze()                # 移除维度为 1 的轴（C=1）
            # 归一化到 0~1（避免数值范围异常导致 viridis 配色失真）
            img_gray_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-6)

            # --------------------------
            # 步骤2：将 Patch 掩码转为「像素级掩码」
            # --------------------------
            patch_mask = mask[idx].detach().cpu().numpy()  # 单个样本的 Patch 掩码：[L,]（如 196）
            patch_mask_2d = patch_mask.reshape(num_patches_per_side, num_patches_per_side)  # 14x14
            # 用 Kronecker 积将 Patch 掩码扩展为像素级掩码（14x14 → 224x224）
            pixel_mask = np.kron(patch_mask_2d, np.ones((patch_size, patch_size)))  # [224, 224]
            # pixel_mask 中：1=被掩码区域，0=保留区域

            # --------------------------
            # 步骤3：生成「viridis 底色 + 纯白掩码区」的图像
            # --------------------------
            # 第一步：将原始灰度图用 viridis 配色转为 RGB 图像（作为掩码图底色）
            # viridis 配色映射：用 plt.cm.viridis 将 0~1 的灰度值转为 (R,G,B)
            img_viridis_rgb = plt.cm.viridis(img_gray_norm)  # [224, 224, 4]（4通道：RGB + Alpha）
            img_viridis_rgb = img_viridis_rgb[..., :3]       # 去掉 Alpha 通道，保留 RGB：[224, 224, 3]

            # 第二步：将「被掩码的像素区域」填充为纯白色 [1,1,1]
            # 找到所有 pixel_mask=1 的位置，将这些位置的 RGB 设为纯白
            img_masked_viridis = img_viridis_rgb.copy()
            img_masked_viridis[pixel_mask == 1] = [1.0, 1.0, 1.0]  # 纯白填充掩码区

            # --------------------------
            # 步骤4：绘制对比图（原图 vs 掩码图）
            # --------------------------
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1行2列，画布大小 14x6

            # 子图1：原始图像（viridis 配色）
            im1 = ax1.imshow(img_gray_norm, cmap='viridis')  # 明确指定 cmap=viridis
            ax1.set_title(f'Original Image (Viridis Colormap)', fontsize=14, pad=10)
            ax1.axis('off')  # 隐藏坐标轴（更简洁）
            # 添加颜色条（可选，帮助解读 viridis 颜色对应的数值）
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Normalized Intensity', fontsize=12)

            # 子图2：掩码图（viridis 底色 + 纯白掩码区）
            im2 = ax2.imshow(img_masked_viridis)  # 已为 RGB 图像，无需再指定 cmap
            ax2.set_title(f'Masked Image (Viridis Base + White Masked Areas)', fontsize=14, pad=10)
            ax2.axis('off')  # 隐藏坐标轴

            # 调整布局，避免标签重叠
            plt.tight_layout()
            # 显示图像（若需要保存，取消下方注释）
            # plt.savefig(f'masked_viridis_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
    model = mae_vit_base_patch16_dec512d8b(in_chans=1)  # in_chans=1 因为输入是灰度图（ST-map）
    
    # 2. 加载输入数据（与你原代码一致）
    # 注意：确保输入数据已归一化（避免数值范围异常）
    stmap_path = r'D:\Code\contactless_rsp\neural_methods\model\stmap_test.npy'
    x_np = np.load(stmap_path)
    x = torch.from_numpy(x_np).reshape(1, 1, 224, 224).float()  # 转张量：[N=1, C=1, H=224, W=224]
    
    # 3. 前向传播，获取 Mask 中间结果
    with torch.no_grad():  # 仅推理，不计算梯度（加快速度）
        loss, BVP_map, mask = model(x, mask_ratio=0.75)  # mask_ratio=0.75 即 75% Patch 被掩码
    
    # 4. 绘制 Mask 结果（关键步骤）
    plot_masked_result_viridis(imgs=x, mask=mask, model=model, num_samples=1)
    
    # （可选）打印关键信息验证
    print(f"Loss: {loss.item():.4f}")
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")  # 应输出 torch.Size([1, 196])（196=14x14 Patch）
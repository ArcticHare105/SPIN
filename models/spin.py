import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange
from .pair_wise_distance import PairwiseDistFunction

def create_model(args):
    return SPIN(colors=args.colors, dim=args.dim, block_num=args.block_num, heads=args.heads, qk_dim=args.qk_dim,\
                mlp_dim=args.mlp_dim, stoken_size=args.stoken_size, upscale=args.upscale)

class SPIN(nn.Module):
    def __init__(self, colors=3, dim=40, block_num=8, heads=1, qk_dim=24, mlp_dim=72, stoken_size=[12, 16, 20, 24, 12, 16, 20, 24], upscale=3):
        super(SPIN, self).__init__()
        self.dim = dim
        self.stoken_size = stoken_size
        self.block_num = block_num
        self.upscale = upscale

        self.first_conv = nn.Conv2d(colors, dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(Block(dim=dim, layer_num=2, stoken_size=[stoken_size[i], stoken_size[i]], heads=heads, qk_dim=qk_dim, mlp_dim=mlp_dim))
            self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        else:
            raise NotImplementedError(
                'Upscale factor is expected to be one of (2, 3, 4), but got {}'.format(upscale))
        self.last_conv = nn.Conv2d(dim, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))

    def forward(self, x):
        """Forward function.
        In traning mode, 'target' should be provided for loss calculation.
        Args:
            x (Tensor): Input image.
            target (Tensor): GT image.
        """
        b, _, h, w = x.size()
        x /= 255.

        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        x = self.first_conv(x)

        for i in range(self.block_num):
            residual = self.blocks[i](x)
            x = x + self.mid_convs[i](residual)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))

        out = base + self.last_conv(out)
        return out * 255.


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        dim (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FFN(nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm = norm_layer(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))


def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels
    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height
    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


def ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6
    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    num_spixels_height = height // sheight
    num_spixels_width = width // swidth
    num_spixels = num_spixels_height * num_spixels_width

    # import pdb; pdb.set_trace()
    # num_spixels_width = int(math.sqrt(num_spixels * width / height))
    # num_spixels_height = int(math.sqrt(num_spixels * height / width))

    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    with torch.no_grad():
        for k in range(n_iter):
            if k < n_iter - 1:

                dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                    / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()
            else:
                dist_matrix = PairwiseDistFunction.apply(
                pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()

    return abs_affinity, num_spixels


class GenSP(nn.Module):
    def __init__(self, n_iter=2):
        super().__init__()
        
        self.n_iter = n_iter
                
    def forward(self, x, stoken_size):
        soft_association, num_spixels = ssn_iter(x, stoken_size, self.n_iter)
        return soft_association, num_spixels


class SPInterAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.sp = nn.Linear(dim, qk_dim, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        head_dim = self.qk_dim // self.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = qk_scale or head_dim ** -0.5

    def forward_stoken(self, x, affinity_matrix):
        x = rearrange(x, 'b c h w -> b (h w) c')
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
        return stokens

    def forward(self, x, affinity_matrix, num_spixels):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        B, C, H, W = x.shape

        x = self.norm(x)

        # generate superpixel stoken
        stoken = self.forward_stoken(x, affinity_matrix) # b, k, c

        # stoken projection
        stoken = self.sp(stoken).permute(0,2,1).reshape(B, self.num_heads, self.qk_dim // self.num_heads, num_spixels) # B, H, C, hh*ww

        # q, k, v projection
        q = self.q(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        k = self.k(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        v = self.v(x).reshape(B, self.num_heads, self.dim // self.num_heads, H*W) # B, H, C, N

        # stoken interaction
        # s_attn = F.normalize(k, dim=-2).transpose(-2, -1) @ F.normalize(stoken, dim=-2) # B, H, H*W, hh*ww
        s_attn = k.transpose(-2, -1) @ stoken * self.scale # B, H, H*W, hh*ww
        s_attn = self.attn_drop(F.softmax(s_attn, -2))
        s_out = (v @ s_attn) # B, H, C, hh*ww

        # x_attn = F.normalize(stoken, dim=-2).transpose(-2, -1) @ F.normalize(q, dim=-2) # B, H, hh*ww, H*W
        x_attn = stoken.transpose(-2, -1) @ q * self.scale
        x_attn = self.attn_drop(F.softmax(x_attn, -2))
        x_out = (s_out @ x_attn).reshape(B, C, H, W) # B, H, C, H*W

        return x_out

class SPIntraAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # superpixels' pixel selection
        _, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk

        q_sp_pixels = torch.gather(q.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        k_sp_pixels = torch.gather(k.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        v_sp_pixels = torch.gather(v.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.dim, -1)) # B, K, C, topk
          
        q_sp_pixels, k_sp_pixels, v_sp_pixels = map(lambda t: rearrange(t, 'b k (h c) t -> b k h t c', h=self.num_heads), \
                                                        (q_sp_pixels, k_sp_pixels, v_sp_pixels)) # b k topk c

        # similarity
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        # attn = F.normalize(q_sp_pixels, dim=-1) @ F.normalize(k_sp_pixels, dim=-1).transpose(-2,-1) # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_sp_pixels # b k h topk c
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')

        out = scatter_mean(v.reshape(B, self.dim, H*W), -1, indices.reshape(B, 1, -1).expand(-1, self.dim, -1), out)
        out = out.reshape(B, C, H, W)

        return out


def scatter_mean(tgt, dim, indices, src):
    count = torch.ones_like(tgt)
    new_src = torch.scatter_add(tgt, dim, indices, src)
    new_count = torch.scatter_add(count, dim, indices, torch.ones_like(src))
    new_src /= new_count

    return new_src


class VanillaAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class LocalTokenAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, stoken_size):
        super().__init__()

        self.stoken_size = stoken_size
        self.norm = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, heads, qk_dim)

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        local_x = []
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                local_x.append(x[:, :, top:down, left:right])
        local_x = torch.stack(local_x, dim=0) 
        local_x = local_x.permute(1, 0, 2, 3, 4).contiguous()
        return local_x

    def local_reverse(self, local_x, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        x_output = torch.zeros_like(x)
        count = torch.zeros((b,h,w), device=x.device)

        index = 0
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                x_output[:, :, top:down, left:right] += local_x[:, index]
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1)
        return x_output

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """

        B, C, H, W = x.shape
        dh, dw = self.stoken_size[0], self.stoken_size[1]

        # import pdb; pdb.set_trace()
        local_x = self.local_partition(x, dh-2, dw-2, dh, dw)
        _, n, _, dh, dw = local_x.shape
        local_x = rearrange(local_x, 'b n c dh dw -> (b n) (dh dw) c')

        local_x = self.attn(self.norm(local_x))
        local_x = rearrange(local_x, '(b n) (dh dw) c  -> b n c dh dw', n=n, dw=dw)
        x = self.local_reverse(local_x, x, dh-2, dw-2, dh, dw)

        return x


class Block(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        heads (int): Head numbers of Attention.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in FFN.
    """
    def __init__(self, dim, layer_num, stoken_size, heads, qk_dim, mlp_dim):
        super(Block,self).__init__()
        self.layer_num = layer_num
        self.stoken_size = stoken_size

        self.gen_super_pixel = GenSP(3)

        self.inter_layer = nn.ModuleList([
                SPInterAttModule(dim, heads, qk_dim),
                FFN(dim, mlp_dim, dim),
            ])
        self.intra_layer = nn.ModuleList([
                SPIntraAttModule(dim, heads, qk_dim, topk=(stoken_size[0]**2)),
                FFN(dim, mlp_dim, dim),
            ])
        self.local_layer = nn.ModuleList([
                LocalTokenAttention(dim, heads, qk_dim, stoken_size),
                FFN(dim, mlp_dim, dim),
            ])


    def forward(self, x):
        # x = self.pos_embed(x)+x

        affinity_matrix, num_spixels = self.gen_super_pixel(x, self.stoken_size)

        inter_attn, inter_ff = self.inter_layer
        x = inter_attn(x, affinity_matrix, num_spixels) + x
        x = inter_ff(x) + x

        intra_attn, intra_ff = self.intra_layer
        x = intra_attn(x, affinity_matrix, num_spixels) + x
        x = intra_ff(x) + x

        local_attn, local_ff = self.local_layer
        x = local_attn(x) + x
        x = local_ff(x) + x

        return x

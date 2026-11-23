"""
Stage 2b building blocks: a Vision Transformer (ViT) and its masked autoencoder (MAE),
written from scratch. Only PyTorch's basic layers are used (Linear, LayerNorm, Conv2d as a
patch cutter); attention is written out by hand below, not taken from a library.

Status: built and checked on CPU (shapes, masking, that the MAE loss falls when it trains on
a few images). NOT yet wired into pretrain.py / train.py, and never trained on real photos.
That needs the laptop GPU and many more unlabeled photos (the roadmap's condition for 2b).

How a ViT reads a photo (Dosovitskiy et al. 2021):
  1. cut the image into a grid of square patches (16 x 16 px: a 512 px photo -> 32 x 32 =
     1,024 patches) and turn each patch into one vector of `dim` numbers (a "token");
  2. add a position code to each token, so the model knows where the patch was (the
     transformer itself has no idea of order or space);
  3. `depth` transformer blocks: in each, every token looks at every other token
     (self-attention) and then is updated on its own by a small 2-layer network (MLP);
  4. average all tokens and map that average to one score, like the ResNet's pooling + head.
Unlike a CNN, nothing forces it to look locally first: it must LEARN that nearby pixels
matter. That needs a lot of data, which is why it gets pretrained (MAE) before disease training.

How MAE pretrains it without labels (He et al. 2022):
  hide a random 75% of the patches; the encoder (the ViT above) sees ONLY the visible 25%,
  which also makes pretraining ~3-4x cheaper; a small decoder gets the encoded visible tokens
  plus a learned "mask token" in every hidden position and redraws the hidden patches' pixels.
  The loss is the mean squared error on the hidden patches only, with each target patch
  normalized to its own mean and spread (MAE's "norm_pix_loss": it makes the model learn
  structure, not overall brightness). Same idea as pretrain.py's SimMIM for the ResNet, but
  MAE drops the hidden patches from the encoder entirely, which a CNN cannot do.

Sizes (ViT-Small, the default): dim 384, depth 12, 6 heads -> ~21.7M weights, about the size
of ResNet-34. Cost grows with the SQUARE of the number of tokens in attention: at 512 px with
16 px patches the full model sees 1,024 tokens; MAE pretraining sees 256 (25%).

    python vit.py          prints the shapes at every step and runs the small CPU checks
"""

import math

import torch
import torch.nn as nn


def sincos_positions(grid, dim):
    """
    Fixed 2-D position codes (as in MAE): half of the `dim` numbers encode the row, half the
    column, each as sines and cosines at different frequencies, like a set of rulers with
    different spacings. Fixed, not learned, so they also work for patches the encoder never sees.
    Returns (grid * grid, dim).
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sine/cosine codes"
    quarter = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    rows, cols = torch.meshgrid(torch.arange(grid, dtype=torch.float64),
                                torch.arange(grid, dtype=torch.float64), indexing="ij")

    def code(pos):                                   # (grid*grid,) -> (grid*grid, dim/2)
        angles = pos.reshape(-1, 1) * freq[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    return torch.cat([code(rows), code(cols)], dim=1).float()


class Attention(nn.Module):
    """
    Multi-head self-attention, written out. For every token:
      query q = "what am I looking for", key k = "what do I contain", value v = "what I pass on".
      weight of token j for token i = softmax over j of (q_i . k_j / sqrt(head size));
      output_i = sum over j of weight_ij * v_j.
    `heads` independent copies run side by side on slices of the vector (each can attend to a
    different kind of relation), then a Linear layer mixes their outputs.
    """

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)           # q, k and v for all heads in one matrix product
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                            # x: (batch, tokens, dim)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # each (batch, heads, tokens, head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (batch, heads, tokens, tokens)
        # softmax in float32: in mixed precision, float16 exponentials overflow easily
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        mixed = (weights @ v).transpose(1, 2).reshape(b, n, d)           # heads side by side again
        return self.out(mixed)


class Block(nn.Module):
    """One transformer block, "pre-norm": x + attention(norm(x)), then x + mlp(norm(x)).
    The `x +` shortcuts play the same role as the ResNet's: deep stacks stay trainable."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def init_weights(module):
    """Small random Linear weights (Xavier), zero biases, LayerNorm as identity: the usual ViT start."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ViT(nn.Module):
    """The encoder, and with its head the disease model: one score per image, like RetinaResNet."""

    def __init__(self, img_size=512, patch=16, dim=384, depth=12, heads=6, dropout=0.2):
        super().__init__()
        assert img_size % patch == 0
        self.patch, self.grid = patch, img_size // patch
        # A Conv2d with kernel = stride = patch size is exactly "cut into patches, then the same
        # Linear layer on each flattened patch", just faster.
        self.embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.register_buffer("pos", sincos_positions(self.grid, dim)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)
        init_weights(self)
        w = self.embed.weight.data                    # MAE: initialize the patch cutter like a Linear layer
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.embed.bias)
        nn.init.normal_(self.head.weight, 0, 0.01)    # same small start as RetinaResNet's head

    def tokens(self, x):
        """(batch, 3, H, W) -> (batch, patches, dim) with position codes added."""
        return self.embed(x).flatten(2).transpose(1, 2) + self.pos

    def encode(self, t):
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward(self, x):
        t = self.encode(self.tokens(x))
        return self.head(self.dropout(t.mean(dim=1))).squeeze(1)     # one raw score per image


class MAE(nn.Module):
    """Masked autoencoder around a ViT encoder. forward(x) -> (loss, predicted patches, mask)."""

    def __init__(self, encoder, mask_ratio=0.75, dec_dim=256, dec_depth=4, dec_heads=8):
        super().__init__()
        self.encoder, self.mask_ratio = encoder, mask_ratio
        dim, p, grid = encoder.head.in_features, encoder.patch, encoder.grid
        self.to_dec = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_positions(grid, dec_dim)[None], persistent=False)
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.to_pixels = nn.Linear(dec_dim, p * p * 3)
        init_weights(nn.ModuleList([self.to_dec, self.dec_blocks, self.dec_norm, self.to_pixels]))
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        """(batch, 3, H, W) -> (batch, patches, patch*patch*3), in the same order as the tokens."""
        b, c, _, _ = x.shape
        p, g = self.encoder.patch, self.encoder.grid
        return x.reshape(b, c, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, p * p * c)

    def unpatchify(self, patches):
        b = patches.shape[0]
        p, g = self.encoder.patch, self.encoder.grid
        return patches.reshape(b, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, g * p, g * p)

    def random_keep(self, b, n, device):
        """
        Per image, which patches stay visible: shuffle the patch numbers with random noise and
        keep the first (1 - mask_ratio). Returns the kept indices (b, n_keep) and the mask
        (b, n): 1 = hidden, 0 = visible.
        """
        n_keep = int(round(n * (1 - self.mask_ratio)))
        order = torch.rand(b, n, device=device).argsort(dim=1)
        keep = order[:, :n_keep]
        mask = torch.ones(b, n, device=device)
        mask.scatter_(1, keep, 0.0)
        return keep, mask

    def forward(self, x):
        t = self.encoder.tokens(x)                                       # (b, n, dim)
        b, n, d = t.shape
        keep, mask = self.random_keep(b, n, x.device)
        visible = torch.gather(t, 1, keep[:, :, None].expand(-1, -1, d))
        encoded = self.encoder.encode(visible)                           # the encoder sees 25% only

        # Decoder input: a mask token everywhere, the encoded tokens put back in their places.
        z = self.to_dec(encoded)
        full = self.mask_token.expand(b, n, -1).clone().to(z.dtype)
        full.scatter_(1, keep[:, :, None].expand(-1, -1, z.shape[2]), z)
        full = full + self.dec_pos
        for block in self.dec_blocks:
            full = block(full)
        pred = self.to_pixels(self.dec_norm(full))                       # (b, n, patch*patch*3)

        target = self.patchify(x)
        mean, var = target.mean(dim=-1, keepdim=True), target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()                   # each patch to its own scale
        per_patch = ((pred.float() - target) ** 2).mean(dim=-1)          # (b, n)
        loss = (per_patch * mask).sum() / mask.sum()                     # hidden patches only
        return loss, pred, mask


def _checks():
    """Shapes and small CPU checks (also run by selftest.py)."""
    torch.manual_seed(0)
    small = dict(img_size=64, patch=16, dim=64, depth=2, heads=4)
    vit = ViT(**small)
    x = torch.randn(3, 3, 64, 64)
    mae = MAE(vit, dec_dim=32, dec_depth=1, dec_heads=4)
    ok = {}
    ok["score shape"] = vit(x).shape == (3,)
    ok["patchify round trip"] = torch.equal(mae.unpatchify(mae.patchify(x)), x)
    loss, pred, mask = mae(x)
    ok["75% hidden per image"] = bool((mask.sum(dim=1) == 12).all()) and pred.shape == (3, 16, 16 * 16 * 3)
    # Attention written out = PyTorch's reference function (checked only, never used).
    att = vit.blocks[0].attn
    t = torch.randn(2, 5, 64)
    qkv = att.qkv(t).reshape(2, 5, 3, 4, 16).permute(2, 0, 3, 1, 4)
    ref = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    ok["attention = reference"] = torch.allclose(att(t), att.out(ref.transpose(1, 2).reshape(2, 5, 64)), atol=1e-5)
    # It learns: 150 steps on 4 fixed smooth images bring the masked-patch loss well down
    # (averages of the first and last 10 steps, since each step hides different patches).
    imgs = torch.nn.functional.interpolate(torch.randn(4, 3, 4, 4), size=64, mode="bilinear", align_corners=False)
    opt = torch.optim.AdamW(mae.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        loss, _, _ = mae(imgs)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    ok[f"MAE loss falls ({first:.3f} -> {last:.3f})"] = last < 0.6 * first
    return ok


if __name__ == "__main__":
    vit = ViT()
    print(f"ViT-Small: {sum(p.numel() for p in vit.parameters()) / 1e6:.1f}M weights "
          f"(ResNet-18 11.2M, ResNet-34 21.3M)")
    mae = MAE(vit)
    print(f"MAE decoder adds {sum(p.numel() for n, p in mae.named_parameters() if not n.startswith('encoder')) / 1e6:.1f}M")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        t = vit.tokens(x)
        print(f"input {tuple(x.shape)} -> tokens {tuple(t.shape)} (32 x 32 patches of 16 px)")
        print(f"score {tuple(vit(x).shape)} <- one score per image")
        loss, pred, mask = mae(x)
        print(f"MAE: encoder sees {int((1 - mask).sum())} of {mask.shape[1]} patches; "
              f"prediction {tuple(pred.shape)}; untrained loss {loss.item():.3f}")
    for name, passed in _checks().items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")

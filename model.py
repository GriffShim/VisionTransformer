import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, img_size, patch_size, chan_num, patch_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.chan_num = chan_num
        self.patch_dim = patch_dim

        self.num_patches = ((img_size // patch_size) ** 2) * chan_num
        self.flattened_size = (patch_size ** 2) * chan_num

        self.to_patch_embedding = nn.Linear(self.flattened_size, self.patch_dim)
        self.un_embed_patch = nn.Linear(self.patch_dim, self.flattened_size)
      
    def forward(self, x):
        x = rearrange(x, 
                      'b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                      s1 = self.patch_size, 
                      s2 = self.patch_size
                      )
        x = self.to_patch_embedding(x)
        return x
    
    def un_patchify(self, x):
        x = self.un_embed_patch(x)
        x = rearrange(x,
                      'b (h w) (s1 s2 c) -> b c (h s1) (w s2)',
                      s1 = self.patch_size,
                      s2 = self.patch_size,
                      h  = self.img_size // self.patch_size,
                      w  = self.img_size // self.patch_size
                      )
        return x

class Attention(nn.Module):
    def __init__(self, dim, dropout = 0):
      super().__init__()
      self.dropout = dropout
      
      self.Q = nn.Linear(dim, dim)
      self.K = nn.Linear(dim, dim)
      self.V = nn.Linear(dim, dim)

      self.scale = dim ** 0.5
    
    def forward(self, x):
      Q = self.Q(x)
      K = self.K(x)

      attention = (Q @ K.transpose(-2, -1)) / self.scale
      attention = F.softmax(attention, dim = -1)

      V = self.V(x)
      attention = attention @ V

      return attention

class FeedForward(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.linear1 = nn.Linear(dim, 4*dim)
    self.linear2 = nn.Linear(4*dim, dim)
    self.act     = nn.GELU()

  def forward(self, x):
    x = self.linear1(x)
    x = self.act(x)
    x = self.linear2(x)
    x = self.act(x)

    return x
  
class PosEnc(nn.Module):
  def __init__(self, t_max, dim):
    super().__init__()
    self.time_steps = t_max
    self.dim = dim
    self.pos_emb = nn.Embedding(t_max, dim)

  def forward(self, x):
    b, t, d = x.shape
    pos = torch.arange(t, device=x.device)
    pos = rearrange(pos, 't -> 1 1 t')
    pos = self.pos_emb(pos)

    x = x + pos

    return x 

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AdaLnBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.ff = FeedForward(dim)
    self.attn = Attention(dim)
    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)
    self.cond_proj = nn.Linear(2 * dim, 6 * dim)

  def forward(self, x, t, c):
    cond = torch.cat([t, c], dim=-1) 
    cond = self.cond_proj(cond) # cond : [B, 6 * C]
    cond = cond.unsqueeze(1)    # cond : [B, 1, 6 * C]
    alpha1, gamma1, beta1, alpha2, gamma2, beta2 = torch.chunk(cond, 6, dim=-1)
    
    x = x + alpha1 * self.attn(modulate(self.ln1(x), beta1, gamma1))
    x = x + alpha2 * self.ff(modulate(self.ln2(x), beta2, gamma2))
    return x
  
class mynet(nn.Module):
    def __init__(self, img_size, patch_size, emb_dim, t_max, attn_num, num_classes):
        super().__init__()
        self.img_size = img_size
        self.patchy = Patchify(img_size = img_size, patch_size = patch_size, chan_num = 3, patch_dim = emb_dim)
        self.t_max = t_max
        self.time_emb = nn.Embedding(t_max, emb_dim)
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.posenc = PosEnc(t_max, emb_dim)
        self.patch_embed = nn.Linear(emb_dim, emb_dim)

        self.blocks = nn.ModuleList([AdaLnBlock(emb_dim) for _ in range(attn_num)])

    def forward(self, x, t, c):
        t = self.time_emb(t) # [B, C]
        c = self.class_emb(c) # [B, C]

        x = self.patchy(x) # x : [B, T, C]

        x = self.posenc(x)                 # [B, T+2, C]
        x = self.patch_embed(x).squeeze(0) # [B, T+2, C]

        for block in self.blocks:
          x = block(x, t, c)
        x = self.patchy.un_patchify(x)
        return x

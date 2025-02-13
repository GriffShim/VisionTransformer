import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import einops
from einops import rearrange

class ddpm:
    def __init__(self, model, beta_1=1e-4, beta_t=2e-2, t_max=400):
        self.model = model
        self.img_size = model.img_size
        self.t_max = t_max
        self.betas = ((beta_t - beta_1) * (1 - torch.cos(torch.arange(0, np.pi/2, (np.pi/2) / t_max))) + beta_1).to('cuda')
        # self.betas = torch.linspace(beta_1, beta_t, t_max).to('cuda')
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def extract(self, const):
        while len(const.shape) < 4:
          const = const.unsqueeze(-1)
        return const.to('cuda')

    def q_sample(self, x_0, t):
        with torch.no_grad():
          x_0 = x_0.to('cuda')
          eps = torch.randn_like(x_0).to('cuda')
          return self.extract(1 - self.alphas_bar[t]).sqrt() * eps + self.extract(self.alphas_bar[t].sqrt()) * x_0, eps

    def p_sample(self, c, w=1, uncond=10):

        self.model.eval()
        x_T = torch.randn(1, 3, self.img_size[0], self.img_size[1]).to('cuda')

        c = torch.tensor(c).unsqueeze(0).to('cuda')
        uncond_label = torch.tensor(uncond).unsqueeze(0).to('cuda')
        with torch.no_grad():
          x_t = x_T
          for t in range(self.t_max-1, -1, -1):
              z = torch.randn_like(x_T).to('cuda') if t > 1 else 0

              t = torch.tensor(t).unsqueeze(0).to('cuda')

              one_over_alpha_sqrt = self.extract(1 / self.alphas[t].sqrt()).to('cuda')
              coef = self.extract(1 - self.alphas[t]) / self.extract(1 - self.alphas_bar[t]).sqrt().to('cuda')
              preds = (1 + w) * self.model(x_t, t, c) - w * self.model(x_t, t, uncond_label)
              x_t = one_over_alpha_sqrt * (x_t - coef*preds).to('cuda')
              x_t = (x_t + self.extract(self.betas[t].sqrt()) * z).to('cuda')

        return x_t

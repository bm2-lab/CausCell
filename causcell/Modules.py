import copy
import torch
import torch.nn as nn
from torch import einsum
from pathlib import Path
import math
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from torch.utils import data
import scanpy as sc
from einops import rearrange, repeat
from .utils import make_beta_schedule, default, exists, extract_into_tensor, BatchedOperation, noise_like
from .utils import create_activation, create_norm, mean_flat, sum_flat, gaussian_parameters
from .utils import timestep_embedding
import torch.nn.functional as F
from typing import Optional
from functools import partial
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import logging
import random
from .Dataset import Dataset

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class DisentanglementEncoder(nn.Module):
    def __init__(self, 
                 profile_size, 
                 out_dim, 
                 num_factor, 
                 causal_dag,
                 label_categories,
                 bias = False,
                 out_act = "gelu",  
                 gamma = 35
                 ):
        super().__init__()
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        self.num_factor = num_factor
        self.out_dim = out_dim
        self.profile_size = profile_size
        self.exogenous_encoder_m_v = nn.Sequential(
            nn.Linear(profile_size, profile_size // 4), 
            Mish(), 
            nn.Linear(profile_size // 4, num_factor * out_dim * 2)
        )
        

        self.causal_dag = nn.Parameter(causal_dag)
        self.causal_dag.requires_grad = False
        self.I = nn.Parameter(torch.eye(num_factor))
        self.I.requires_grad = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_factor))
        else:
            self.register_parameter('bias', None)

        self.label_predictor = nn.ModuleList()
        for idx, num in enumerate(label_categories):
            self.label_predictor.append(nn.Sequential(
                nn.Linear(out_dim, num),
                nn.Softmax(dim = 1) 
            )
            )

        self.multilabelmulticate_loss = nn.CrossEntropyLoss()

        
        # discriminator for o and v
        self.discriminator_ov = nn.Linear(out_dim, 1)
        self.discriminator_ov2 = nn.Linear(num_factor, 1)
        self.discriminator_ov_act = nn.Sigmoid()
        
        self.gamma = gamma
    
    
    def mask_z(self, x):
        
        x = torch.matmul(self.causal_dag, x)
        
        return x
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def calculat_prior_kl(self, mean, log_var):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        """
        batch_size = mean.shape[0]
        kl_prior = self.normal_kl(
            mean1=mean, logvar1=log_var, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
        # return sum_flat(kl_prior) / np.log(2.0)

    def sample(self, mean, log_var):
        noise = torch.randn_like(mean)
        return mean + (0.5 * log_var).exp() * noise
    
    def forward(self, x, o):
        exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v(x), self.num_factor * self.out_dim, dim=-1)
        prior_kl = self.calculat_prior_kl(exogenous_factor_m, exogenous_factor_v).mean()
        
        exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
        exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)

        z = torch.inverse(self.I - self.causal_dag).matmul(exogenous_embs)
        
        concept_embs = z

        m_concept_embs = self.mask_z(concept_embs) + exogenous_embs
        mask_recon_loss = ((concept_embs - m_concept_embs) ** 2).mean()
        
        pred_o = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_o.append(predictor(concept_embs[:,idx,:]))
        pred_o_loss = 0
        for idx, pred_o_idx in enumerate(pred_o):
            pred_o_loss_idx = self.multilabelmulticate_loss(pred_o_idx, o[:,idx])
            pred_o_loss += pred_o_loss_idx
            # print(pred_o_loss_idx)
        
        # take mean-level loss
        pred_o_loss /= idx + 1
        
        # new adversirial part
        pred_u = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_u.append(predictor(concept_embs[:, -1, :]))
        pred_u_loss = 0
        for idx, pred_u_idx in enumerate(pred_u):
            pred_u_loss_idx = self.multilabelmulticate_loss(pred_u_idx, o[:,idx])
            pred_u_loss += pred_u_loss_idx
        # take mean-level loss
        discriminator_loss = - pred_u_loss / (idx + 1)
        
        # take sum-level loss
        # discriminator_loss = - pred_u_loss
        
        return concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def extract_exogenous_embs(self, x):
        with torch.no_grad():
            exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v.eval()(x), self.num_factor * self.out_dim, dim=-1)
            exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
            exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)
        return exogenous_embs

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim, 
                 context_dim, 
                 heads = 8, 
                 dim_head = 64, 
                 dropout = 0., 
                 qkv_bias = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, *, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mnv = max_neg_value(sim) - torch.finfo(sim.dtype).max
            if sim.shape[1:] == sim.shape[1:]:
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            else:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, mnv)
        
        attn = sim.softmax(dim = -1)
        # print(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        d_head: int = 64, 
        self_attn: bool = False,
        cross_attn: bool = True,
        ts_cross_attn: bool = False, 
        final_act: Optional[nn.Module] = None,
        dropout: float = 0, 
        context_dim: Optional[int] = None, 
        gated_ff: bool = True, 
        checkpoint: bool = False,
        qkv_bias: bool = False, 
        linear_attn: bool = False, 
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu = gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim = dim, 
                context_dim = context_dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim = dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, "Checkpointing not available yet"
    
    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x


class Denoise_net(nn.Module):
    def __init__(self, 
                 dim, 
                 out_dim, 
                 num_factor, 
                 causal_dag, 
                 label_categories, 
                 depth = 4,
                 num_heads = 4, 
                 dim_head = 64,
                 dropout = 0., 
                 norm_type = "layernorm", 
                 num_layers = 1, 
                 act = 'gelu', 
                 out_act = None, 
                 with_time_emb = True):
        super().__init__()
        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), 
                nn.Linear(dim, dim * 4), 
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim),
                act,
                create_norm(norm_type, dim),
                nn.Dropout(dropout)
            ))
        self.layers.append(nn.Sequential(nn.Linear(dim, out_dim), out_act))
        
        # disentanglement encoder
        self.DisentanglementEncoder = DisentanglementEncoder(dim, 32, num_factor, causal_dag, label_categories)
        self.Cross_attention_module = nn.ModuleList([
            BasicTransformerBlock(out_dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=32, 
                                  qkv_bias=True, dropout=dropout, final_act=None)
            for _ in range(depth)
        ])
        self.decoder_norm = create_norm(norm_type, out_dim)
        
    def forward(self, x, x_start, time, labels=None, concept_embs=None):
        # if self.cond_embed is not None:
        #     cond_emb = self.cond_embed(conditions)[0]
        #     x = x + cond_emb.squeeze(1)
        if labels is not None and concept_embs is None:
            concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = self.DisentanglementEncoder(x_start, labels)
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            x = x + t
            x = x.unsqueeze(1)
            for blk in self.Cross_attention_module:
                x = blk(x = x, context = concept_embs)
            x.squeeze_(1)
            x = self.decoder_norm(x)
            for layer in self.layers:
                x = layer(x)
            # x = self.layers(x)
            
            return x, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
        
        elif labels is None and concept_embs is None:
            # print("No cross attention generation")
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            x = x + t
            x = self.decoder_norm(x)
            for layer in self.layers:
                x = layer(x)
            # x = self.layers(x)
            return x
        
        elif labels is None and concept_embs is not None:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            x = x + t
            x = x.unsqueeze(1)
            for blk in self.Cross_attention_module:
                x = blk(x = x, context = concept_embs)
            x.squeeze_(1)
            x = self.decoder_norm(x)
            for layer in self.layers:
                x = layer(x)
            # x = self.layers(x)
            return x
        
        else:
            print("No condition for labels and factor embs all exisits")
            return

class GaussianDiffusion(nn.Module):
    def __init__(self, 
                 denosie_fn, 
                 *, 
                 profile_size, 
                #  channels = 3, 
                 timesteps = 1000, 
                 loss_type = "l1", 
                 betas = None):
        super().__init__()
        self.profile_size = profile_size
        self.denosie_fn = denosie_fn
        
     
        
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = make_beta_schedule("linear", timesteps)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)   
          
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

 
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.loss_type = loss_type
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
    
    def q_mean_variance(self, x_start, t):
        """
        Given x_0 and t, output x_t by adding noise
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        
        """
        assert x_t.shape == noise.shape, "Please check the code and data"
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
                )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool):
        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, t))
        x_recon = self.denosie_fn(x, t)
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        
        b = shape[0]
        img = torch.randn(shape, device = device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop((batch_size, profile_size))

    def p_mean_variance_with_factor(self, x, t, concept_embs, clip_denoised: bool, eps = False):
        
        x_start = None
        if eps:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, x_start, t, concept_embs = concept_embs))
        else:
            x_recon = self.denosie_fn(x, x_start, t, concept_embs = concept_embs)
            
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_with_factor(self, x, t, concept_embs, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_with_factor(x=x, t=t, concept_embs=concept_embs, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_with_factor(self, shape, concept_embs):
        device = self.betas.device
        
        b = shape[0]
        img = torch.randn(shape, device = device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample_with_factor(img, torch.full((b,), i, device=device, dtype=torch.long), concept_embs)
        return img

    @torch.no_grad()
    def sample_with_factor(self, concept_embs, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop_with_factor((batch_size, profile_size), concept_embs)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        
        assert x1.shape == x2.shape
        
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        
        img = (1 - lam) * xt1 + lam *xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        
        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +  
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
                )
            
    def p_losses(self, x_start, t, labels, weights, noise = None, eps = False):
        b, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = self.denosie_fn(x_noisy, x_start, t, labels)
        
        assert x_recon.shape == x_noisy.shape, "Please check the code and data"
        
        if self.loss_type == "l1":
            if eps:
                loss = (((noise - x_recon).abs()) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon).abs()) * weights[:, None]).sum()
        elif self.loss_type == "l2":
            if eps:
                loss = (((noise - x_recon)**2) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon)**2) * weights[:, None]).sum()
        else:
            raise NotImplementedError()
        
        return loss, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def forward(self, x, *args, **kwargs):
        b, c, device, profile_size, = *x.shape, x.device, self.profile_size
        assert c == profile_size, f'dimension of gene expression profile must be {profile_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

class Trainer(object):
    def __init__(
        self, 
        diffusion_model, 
        folder, 
        factor_list,
        *, 
        ema_decay = 0.995, 
        profile_size = 200, 
        train_batch_size = 32, 
        train_lr = 2e-5, 
        train_num_steps = 100000, 
        gradient_accumulate_every = 2, 
        fp16 = False, 
        step_start_ema = 2000, 
        update_ema_every = 1000, 
        save_and_sample_every = 10000,
        results_folder = './results',
        train_log=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        
        self.batch_size = train_batch_size
        self.profile_size = diffusion_model.profile_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.ds = Dataset(folder, profile_size, factor_list)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)
        
        self.step = 0
        
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level = 'O1')
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        
        if train_log:
            self.logger = get_logger(results_folder + '/training.log')
        self.train_log = train_log
        self.reset_parameters()
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def save(self, milestone):
        data = {
            'step': self.step, 
            'model': self.model.state_dict(), 
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        
    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        
        while self.step <= self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                loss_recon, mask_recon_loss, loss_pred_o, loss_discriminator, prior_kl = self.model(data[0].cuda(), data[1].cuda(), data[2].cuda())
                loss = loss_recon + loss_pred_o + loss_discriminator + prior_kl
                # loss = loss_recon
                # loss = loss_pred_o
                if self.train_log:
                    self.logger.info(f'{self.step}:{i}\tloss_recon:{loss_recon.item()}\tloss_pred_o:{loss_pred_o.item()}\tloss_discriminator:{loss_discriminator.item()}\tprior_kl:{prior_kl.item()}')
                # self.logger.info(f'{self.step}:{i}\tloss_recon:{loss_recon.item()}\tloss_pred_o:{loss_pred_o.item()}\tprior_kl:{prior_kl.item()}')
                # print(f'{self.step}:{loss_recon.item()}')
                # print(f'{self.step}:{loss_pred_o.item()}')
                # print(f'{self.step}:{loss_discriminator.item()}')
                # print(f'{self.step}:{prior_kl.item()}')
                # print(f'{self.step}:{loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)
            
            self.opt.step()
            self.opt.zero_grad()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
            
            self.step += 1
        
        print('training completed')

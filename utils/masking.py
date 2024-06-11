import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)

    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            if C:
                # For a continuous timestamps, mask a random selection of channels
                num_channels_to_mask = np.random.randint(1, C + 1) # Randomly decide how many channels to mask
                index = np.random.choice(C, num_channels_to_mask, replace=False) # Select random channels to mask
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t + l] = False
    return res


def expand_tensor(input_tensor, third_dim_size):
    # 将输入张量转换为三维张量
    expanded_tensor = input_tensor.unsqueeze(2).expand(-1, -1, third_dim_size)

    return expanded_tensor.bool()


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def generate_geometric_mask(B, T, C=None, p=0.75, l=3):
    if C:
        mask = np.ones((B, T, C), dtype=bool)
    else:
        mask = np.ones((B, T), dtype=bool)

    for i in range(B):
        if C:
            for c in range(C):
                mask[i, :, c] = geom_noise_mask_single(T, l, p)
        else:
            mask[i, :] = geom_noise_mask_single(T, l, p)

    return torch.from_numpy(mask).to(torch.bool)


def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, 1 - p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, 1 - p, size=(B, T))).to(torch.bool)


def patch_mask(x, mask_ratio, patch_len=12, stride=12):
    px = x.clone().permute(0, 2, 1)

    padding_patch_layer = nn.ReplicationPad1d((0, stride))
    px = padding_patch_layer(px)
    px = px.unfold(dimension=-1, size=patch_len, step=stride)
    px = torch.reshape(px, (px.shape[0] * px.shape[1], px.shape[2], px.shape[3]))

    mask = generate_binomial_mask(px.size(0), px.size(1), p=mask_ratio).to(x.device)

    return mask

def mask_function(x, args):
    b, s, c = x.shape

    if args.masked_rule == 'binomial':
        mask = generate_binomial_mask(x.size(0), x.size(1), p=args.mask_rate).to(x.device)
        mask = expand_tensor(mask, x.shape[-1])
    elif args.masked_rule == 'channel_binomial':
        mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=args.mask_rate).to(x.device)
    elif args.masked_rule == 'continuous':
        mask = generate_geometric_mask(x.size(0), x.size(1), p=args.mask_rate, l=args.lm).to(x.device)
        mask = expand_tensor(mask, x.shape[-1])
    elif args.masked_rule == 'channel_continuous':
        mask = generate_geometric_mask(x.size(0), x.size(1), x.size(2), p=args.mask_rate, l=args.lm).to(x.device)
    elif args.masked_rule == 'mask_last':
        mask = x.new_full((x.size(0), x.size(1), x.size(2)), True, dtype=torch.bool)
        idx = int(x.shape[1] * args.mask_rate)
        mask[:, -idx:, :] = False
    elif args.masked_rule == 'mask_patch':
        mask = patch_mask(x, args.mask_rate, args.patch_len, args.stride)
        mask = expand_tensor(mask, args.patch_len)
        mask = mask.reshape(b, c, -1)[:, :, :s].permute(0, 2, 1)
    else:
        raise ValueError(f'\'{args.mask_rate}\' is a wrong argument for mask function!')

    x_ = mask * x

    return x, x_, mask  # x: [b, s, c] unmasked: True, masked, False


# MAE Masking
def random_masking(xb, mask_ratio=0.75):

    bs, L, nvars, D = xb.shape  # xb: [bs x num_patch x n_vars x patch_len]
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))  # x_masked: [bs x num_patch x nvars x patch_len]


    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    mask = mask.permute(0, 2, 1)
    mask = mask.reshape(-1, L) # [bs * nvars x num_patch]

    return x_masked, x_kept, mask, ids_restore


def glm_geometric_mask(patch_x_emb, patch_x, p=0.5, l=3):
    """
    :param patch_x_emb: bs x c x patch_num x d_model
    :param patch_x: bs x c x patch_num x patch_len
    :param p:
    :param l:
    :return:
    """
    B, C, L, D = patch_x_emb.shape

    patch_x_emb_sort = torch.zeros_like(patch_x_emb).to(patch_x_emb.device)
    patch_x_sort = torch.zeros_like(patch_x).to(patch_x.device)

    mask = np.ones((B, C, L), dtype=bool)
    mask_sort = np.ones((B, C, L), dtype=bool)


    for i in range(B):
        for c in range(C):
            mk = geom_noise_mask_single(L, l, p)

            mask[i, c, :] = mk
            mask_sort[i, c, :] = np.sort(mk)[::-1]

            patch_x_emb_sort[i, c, :] = torch.cat((patch_x_emb[i, c, mk], patch_x_emb[i, c, ~mk]), dim=0)
            patch_x_sort[i, c, :] = torch.cat((patch_x[i, c, mk], patch_x[i, c, ~mk]), dim=0)

    mask = torch.from_numpy(mask).to(patch_x_emb.device)  # B x C x L
    mask_sort = torch.from_numpy(mask_sort).to(patch_x_emb.device)  # B x C x L x D .unsqueeze(-1).repeat(1, 1, 1, D)

    return patch_x, mask, patch_x_emb_sort, mask_sort, patch_x_sort


# MAE Masking
def random_masking_v2(xb, mask_ratio=0.75):

    bs, L, D = xb.shape  # [bs x n_vars x d_model]
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x d_model]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x d_model]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x d_model]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x L x d_model]


    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x L]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]

    return x_masked, x_kept, mask, ids_restore


def patch_random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConToMe (Context Token Merging)
Aggregates background tokens across frames into a unified background summary.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from typing import Callable, Tuple, List, Union


# MPS gather workaround
def mps_gather_workaround(input, dim, index):
    index = index.long()
    if input.shape[-1] == 1:
        return torch.gather(input.unsqueeze(-1),
                            dim - 1 if dim < 0 else dim,
                            index.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def get_objective_score(score_attn):
    """Compute saliency score from attention."""
    score_attn = score_attn.mean(dim=1)
    scores = (score_attn * torch.log(score_attn + 1e-6)).sum(dim=2).unsqueeze(-1)

    # foreground removal
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True) + 1e-6)
    score_mask = scores >= scores.mean(dim=1, keepdim=True)

    # background sharpening
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True) + 1e-6)
    scores[score_mask] = 0.0
    return scores


def gather_(input: torch.Tensor, dim: int, index: torch.Tensor):
    if index is None:
        raise ValueError("index can't be None in gather_")
    index = index.long()
    if input.device.type == "mps":
        return mps_gather_workaround(input, dim, index)
    else:
        return torch.gather(input, dim=dim, index=index)


# ---------------------------------------------------------------------------
# Main ConToMe function
# ---------------------------------------------------------------------------
def ConToMe(x, attn, info, layer, K, V):
    """
    Merge all background tokens across frames (batch) into a unified summary.
    Now supports iterative merging with source tracking.

    Args:
        x: [B, T, C]
        attn: [B, heads, T, T]
        info: config dict with r_merge ratio, etc.
        layer: current layer index
        K, V: [B, heads, T, head_dim]

    Returns:
        fg_tokens [B, T_fg, C]
        merged_bg_tokens [1, T_bg_merged, C]
        K_merged, V_merged [1, T_bg_merged, head_dim]
    """
    B, T, C = x.shape
    device = x.device

    # ----------------------------------------------------------
    # 1. Compute saliency (background mask)
    # ----------------------------------------------------------
    score_obj = get_objective_score(attn)
    bg_mask = (score_obj.squeeze(-1) == 0)  # True for background

    # ----------------------------------------------------------
    # 2. Extract fg/bg tokens per frame
    # ----------------------------------------------------------
    idx_bg = [torch.nonzero(bg_mask[b], as_tuple=False).squeeze(-1) for b in range(B)]
    idx_fg = [torch.nonzero(~bg_mask[b], as_tuple=False).squeeze(-1) for b in range(B)]

    bg_tokens_per_batch = [x[b, idx, :] for b, idx in enumerate(idx_bg)]
    fg_tokens_per_batch = [x[b, idx, :] for b, idx in enumerate(idx_fg)]

    # Pad fg_tokens for return (we need per-frame structure)
    fg_tokens = pad_sequence(fg_tokens_per_batch, batch_first=True, padding_value=0.0)

    # ----------------------------------------------------------
    # 3. Average K/V across heads and extract bg 
    # ----------------------------------------------------------
    K = K.mean(dim=1)  # [B, T, head_dim]
    V = V.mean(dim=1)  # [B, T, head_dim]

    # Extract K/V for bg tokens (list of tensors with different lengths)
    K_bg_per_batch = [K[b, idx, :] for b, idx in enumerate(idx_bg)]
    V_bg_per_batch = [V[b, idx, :] for b, idx in enumerate(idx_bg)]

    # ----------------------------------------------------------
    # 4. Flatten background tokens across batch 
    # ----------------------------------------------------------
    # This avoids introducing zeros that would corrupt the cache
    bg_tokens_flatten = torch.cat(bg_tokens_per_batch, dim=0).unsqueeze(0)  # [1, sum(T_bg_i), C]
    K_bg_flatten = torch.cat(K_bg_per_batch, dim=0).unsqueeze(0)            # [1, sum(T_bg_i), head_dim]
    V_bg_flatten = torch.cat(V_bg_per_batch, dim=0).unsqueeze(0)            # [1, sum(T_bg_i), head_dim]

    T_bg_total = bg_tokens_flatten.shape[1]
    
    # Store original bg_tokens before any merging
    bg_tokens_original = bg_tokens_flatten.clone()
    
    # Initialize source tracking: each token starts with itself
    source_map = [[i] for i in range(T_bg_total)]
    
    if T_bg_total % 2 != 0:
        # Make even number of tokens for pairwise merge
        pad_bg = torch.zeros(1, 1, C, device=device)
        pad_kv = torch.zeros(1, 1, K_bg_flatten.shape[-1], device=device)
        bg_tokens_flatten = torch.cat([bg_tokens_flatten, pad_bg], dim=1)
        K_bg_flatten = torch.cat([K_bg_flatten, pad_kv], dim=1)
        V_bg_flatten = torch.cat([V_bg_flatten, pad_kv], dim=1)
        T_bg_total += 1
        source_map.append([T_bg_total - 1])  # Padding token sources itself

    # ----------------------------------------------------------
    # 5. Token merging with source tracking (AGGRESSIVE - single pass)
    # ----------------------------------------------------------
    # Single merge pass with aggressive compression for better FLOP savings
    # Merge ~50% of background tokens (similar to ToMe)
    r_merge_bg = max(T_bg_total // 2, 1)  # Merge 50% of pairs
    
    merge = merging(bg_tokens_flatten, r_merge=r_merge_bg, score_obj=None)

    # create matching size tensor to track cluster sizes
    size = torch.ones_like(bg_tokens_flatten[..., 0, None])

    # Merge with source tracking
    merged_bg_tokens, source_map = merge(
        bg_tokens_flatten * (size / size.amax(dim=-2, keepdim=True)),
        mode="sum",
        track_source=True,
        source_map=source_map
    )
    size = merge(size, mode="sum")
    merged_bg_tokens = merged_bg_tokens / (size / size.amax(dim=-2, keepdim=True))
    
    K_merged = merge(K_bg_flatten, mode="sum")
    V_merged = merge(V_bg_flatten, mode="sum")
    
    # Store source mapping in info
    info["source"] = source_map
    info["bg_tokens_original"] = bg_tokens_original
    info["size"] = size
    
    print(f"    Layer {layer}: BG tokens {T_bg_total} → {merged_bg_tokens.shape[1]} (single merge pass)")
    
    return fg_tokens, merged_bg_tokens, K_merged, V_merged


# ---------------------------------------------------------------------------
# Helper: merging
# ---------------------------------------------------------------------------
def merging(metric: torch.Tensor, r_merge: int, score_obj: torch.Tensor):
    """
    Build merge function for token grouping.
    Operates on flattened [1, T, C] tensor.
    
    Returns a merge function that can also track source token mappings.
    """
    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = (a @ b.transpose(-1, -2) + 1) / 2  # cosine similarity in [0,1]

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r_merge:, :]
        src_idx = edge_idx[..., :r_merge, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="sum", dtype=torch.float32, track_source=False, source_map=None):
        """
        Args:
            x: tensor to merge
            mode: reduction mode ('sum' or 'amax')
            dtype: computation dtype
            track_source: if True, also update and return source_map
            source_map: list where source_map[i] = list of original token indices merged into token i
        
        Returns:
            merged_x (and optionally updated source_map if track_source=True)
        """
        ori_dtype = x.dtype
        x = x.to(dtype=dtype)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, _, c = src.shape
        
        # gather unmerged and merged
        unm = src.gather(dim=-2, index=unm_idx.expand(n, unm_idx.shape[1], c))
        src = src.gather(dim=-2, index=src_idx.expand(n, src_idx.shape[1], c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, dst_idx.shape[1], c),
                                 src, reduce=mode)
        x = torch.cat([unm, dst], dim=-2)
        
        if track_source and source_map is not None:
            # Update source mapping
            # Split source_map into src (even) and dst (odd) indices
            src_sources = [source_map[i*2] for i in range(len(source_map)//2)]
            dst_sources = [source_map[i*2 + 1] for i in range(len(source_map)//2)]
            
            # Unmerged sources (from src side, based on unm_idx)
            new_source_map = []
            unm_idx_list = unm_idx[0, :, 0].cpu().tolist()
            for idx in unm_idx_list:
                new_source_map.append(src_sources[idx])
            
            # Merged sources (dst absorbs src based on src_idx -> dst_idx)
            src_idx_list = src_idx[0, :, 0].cpu().tolist()
            dst_idx_list = dst_idx[0, :, 0].cpu().tolist()
            
            # Create merged sources for dst tokens
            dst_merged_sources = [list(dst_sources[i]) for i in range(len(dst_sources))]
            for src_i, dst_i in zip(src_idx_list, dst_idx_list):
                # Merge src_sources[src_i] into dst_merged_sources[dst_i]
                dst_merged_sources[dst_i].extend(src_sources[src_i])
            
            # Append merged dst sources to result
            new_source_map.extend(dst_merged_sources)
            
            return x.to(dtype=ori_dtype), new_source_map
        
        return x.to(dtype=ori_dtype)

    return merge


# ---------------------------------------------------------------------------
# merge_wavg
# ---------------------------------------------------------------------------
def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor,
               source_trace: int = 0, source: list = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    size_max = size.amax(dim=-2, keepdim=True)
    
    # Merge tokens
    if source_trace and source is not None:
        x_merged, source = merge(x * (size / size_max), mode="sum", track_source=True, source_map=source)
        size = merge(size, mode="sum")
    else:
        x_merged = merge(x * (size / size_max), mode="sum")
        size = merge(size, mode="sum")
    
    x = x_merged / (size / size_max)
    return x, size, source


# ---------------------------------------------------------------------------
# compute_bg_mask_inference
# ---------------------------------------------------------------------------
def compute_bg_mask_inference(x, attn, bg_tokens_cached, K_cached, V_cached, thresh=0.5):
    """
    Find matching between current frame's background tokens and cached background tokens.
    When multiple test tokens match the same cached token, they are all replaced by that single cached token.

    Args:
        x: [B, T, C] current frame tokens
        attn: attention map for current layer
        bg_tokens_cached: [1, T_cached, C] cached merged background tokens
        K_cached: [1, num_heads, T_cached, head_dim] cached K values
        V_cached: [1, num_heads, T_cached, head_dim] cached V values
        thresh: similarity threshold for matching

    Returns:
        dict with:
            - 'matched_indices': list of token indices in current frame that match cache
            - 'cache_kv_indices': corresponding indices in the cache
            - 'cached_K': K_cached
            - 'cached_V': V_cached
            - 'similarity_stats': dict with max/min/mean similarity
            - 'deduplicated_matches': dict mapping cache_idx -> list of test token indices that matched it
        OR None if no tokens match
    """
    B, T, _ = x.shape
    score_obj = get_objective_score(attn)
    bg_mask = (score_obj.squeeze(-1) == 0)

    # Extract background tokens per batch
    idx_bg = [torch.nonzero(bg_mask[b], as_tuple=False).squeeze(-1) for b in range(B)]
    bg_tokens_per_batch = [x[b, idx, :] for b, idx in enumerate(idx_bg)]
    bg_tokens = pad_sequence(bg_tokens_per_batch, batch_first=True, padding_value=0.0)  # [B, T_bg, C]

    # Normalize for cosine similarity
    a = bg_tokens / (bg_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    b = bg_tokens_cached / (bg_tokens_cached.norm(dim=-1, keepdim=True) + 1e-6)

    # Cosine similarity in [0,1]
    scores = (a @ b.transpose(-1, -2) + 1) / 2  # [B, T_bg, T_cached]
    sim_val, sim_idx = scores.max(dim=-1)       # [B, T_bg]

    # Find which background tokens exceed threshold and map to original indices
    matched_indices = []  # Indices in current frame (original token positions)
    cache_kv_indices = []  # Corresponding indices in cache
    
    # Track which test tokens match which cached tokens
    cache_to_test_tokens = {}  # cache_idx -> [list of test token indices]
    
    for b in range(B):
        matched = (sim_val[b] > thresh).nonzero(as_tuple=False).squeeze(-1)
        if matched.numel() > 0:
            for m in matched:
                if idx_bg[b].numel() > m:
                    # Original token position in current frame
                    token_idx = idx_bg[b][m].item()
                    # Corresponding cache index
                    cache_idx = sim_idx[b, m].item()
                    
                    matched_indices.append(token_idx)
                    cache_kv_indices.append(cache_idx)
                    
                    # Track reverse mapping
                    if cache_idx not in cache_to_test_tokens:
                        cache_to_test_tokens[cache_idx] = []
                    cache_to_test_tokens[cache_idx].append(token_idx)
    
    # Similarity statistics
    similarity_stats = {
        'max': sim_val.max().item(),
        'min': sim_val.min().item(),
        'mean': sim_val.mean().item(),
        'num_bg_tokens': bg_tokens.shape[1],
        'num_cached_tokens': bg_tokens_cached.shape[1]
    }
    
    if len(matched_indices) > 0:
        cache_info = {
            'matched_indices': matched_indices,
            'cache_kv_indices': cache_kv_indices,
            'cached_K': K_cached,
            'cached_V': V_cached,
            'similarity_stats': similarity_stats,
            'deduplicated_matches': cache_to_test_tokens  # NEW: track multiple matches
        }
    else:
        cache_info = None
    
    return cache_info
    
    
    
    
    
    
    
    
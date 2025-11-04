"""
CORRECT Performance Comparison: ViT with K/V Caching Pipeline

This script properly implements:
1. Build K/V cache from N-1 frames using ConToMe
2. Run first 6 transformer blocks normally on all cache frames
3. Use attention from layer 5 to identify background tokens
4. Merge background tokens across all frames using ConToMe
5. For test frame: run blocks 0-5 normally
6. Use compute_bg_mask_inference with threshold to find cached token matches
7. Reuse cached K/V for matched background tokens in blocks 6-11
8. Compare performance with different thresholds (0.5 to 0.95)

Architecture:
- Blocks 0-5: Run normally (no caching)
- Blocks 6-11: Use K/V cache for matched background tokens

Threshold explanation:
- 0.95: Very strict matching, almost no tokens cached (minimal speedup)
- 0.5: Loose matching, more tokens cached (more speedup but potentially lower accuracy)
"""
import torch
import timm
from PIL import Image
import os
import requests
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ============================================================================
# Accuracy Calculation Functions
# ============================================================================

def calculate_top_k_accuracy(predictions, ground_truths, k=1):
    """
    Calculate Top-K accuracy
    
    Args:
        predictions: List of predicted class indices (or lists of top-k indices)
        ground_truths: List of ground truth class indices
        k: Top-K to consider
    
    Returns:
        accuracy: Float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) must have same length")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if isinstance(pred, (list, tuple)):
            # pred is already top-k indices
            if gt in pred[:k]:
                correct += 1
        else:
            # pred is single index
            if k == 1:
                if pred == gt:
                    correct += 1
            else:
                # This shouldn't happen - need top-k predictions
                raise ValueError("For k>1, predictions must be lists of top-k indices")
    
    return correct / len(predictions)

def get_top_k_predictions(logits, k=5):
    """
    Get top-k predicted class indices from logits
    
    Args:
        logits: Tensor of shape [B, num_classes]
        k: Number of top predictions to return
    
    Returns:
        top_k_indices: List of lists containing top-k class indices
    """
    probs = torch.softmax(logits, dim=-1)
    top_k_vals, top_k_indices = torch.topk(probs, k, dim=-1)
    return top_k_indices.cpu().tolist()
from ConToMe import ConToMe, compute_bg_mask_inference, get_objective_score

def count_attention_flops(B, num_heads, T, head_dim):
    """Count FLOPs for attention computation"""
    qk_flops = B * num_heads * T * T * head_dim
    av_flops = B * num_heads * T * T * head_dim
    return qk_flops + av_flops

def run_attention_block(block, x, cache_info=None, merge_ratio=0.5):
    """
    Run one attention block with optional token merging for sequence reduction
    
    AGGRESSIVE STRATEGY: 
    1. Merge matched BG tokens (that point to same cache token)
    2. ALSO merge unmatched tokens using ToMe-style bipartite matching
    
    This progressively reduces sequence through cached layers
    
    Args:
        merge_ratio: Fraction of tokens to merge (0.5 = reduce by ~25%)
    
    Returns: output, attn, k, v, flops_used
    """
    B, T, C = x.shape
    num_heads = block.attn.num_heads
    head_dim = C // num_heads
    
    # Layer norm
    normed = block.norm1(x)
    
    if cache_info is None:
        # Standard path: compute Q,K,V for all tokens
        qkv = block.attn.qkv(normed)
        qkv = qkv.reshape(B, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        tokens_kv_computed = T
    else:
        # PHASE 1: Merge matched BG tokens
        high_conf_matches = cache_info.get('high_confidence_matches', {})
        medium_conf_matches = cache_info.get('medium_confidence_matches', {})
        
        # Combine all matches for merging
        all_matches = {}
        for cache_idx, test_indices in high_conf_matches.items():
            all_matches[cache_idx] = test_indices
        for cache_idx, test_indices in medium_conf_matches.items():
            if cache_idx in all_matches:
                all_matches[cache_idx].extend(test_indices)
            else:
                all_matches[cache_idx] = test_indices
        
        # Track which test tokens are matched
        matched_test_indices = set()
        for test_indices in all_matches.values():
            matched_test_indices.update(test_indices)
        
        # Build intermediate sequence by merging matched tokens
        intermediate_tokens = []
        token_is_merged = {}  # Maps position in intermediate -> is_merged_group
        old_to_intermediate = {}
        intermediate_idx = 0
        
        processed_groups = set()
        for old_idx in range(T):
            if old_idx in matched_test_indices:
                # Find which cache token this matches
                cache_idx = None
                for c_idx, test_idxs in all_matches.items():
                    if old_idx in test_idxs:
                        cache_idx = c_idx
                        break
                
                if cache_idx not in processed_groups:
                    # First token in this group: merge all test tokens
                    group_test_indices = all_matches[cache_idx]
                    group_tokens = [x[:, idx:idx+1, :] for idx in group_test_indices]
                    
                    # Merge test tokens using average
                    merged_token = torch.mean(torch.cat(group_tokens, dim=1), dim=1, keepdim=True)
                    intermediate_tokens.append(merged_token)
                    token_is_merged[intermediate_idx] = True
                    
                    # Map all tokens in group to this position
                    for idx in group_test_indices:
                        old_to_intermediate[idx] = intermediate_idx
                    
                    processed_groups.add(cache_idx)
                    intermediate_idx += 1
            elif old_idx not in old_to_intermediate:
                # Unmatched token: keep for now
                intermediate_tokens.append(x[:, old_idx:old_idx+1, :])
                token_is_merged[intermediate_idx] = False
                old_to_intermediate[old_idx] = intermediate_idx
                intermediate_idx += 1
        
        # PHASE 2: Progressive merging of remaining tokens (ToMe-style)
        # Apply additional merging to reduce sequence further
        x_intermediate = torch.cat(intermediate_tokens, dim=1)
        T_intermediate = x_intermediate.shape[1]
        
        # Calculate how many more tokens to merge
        additional_merge_ratio = cache_info.get('merge_ratio', merge_ratio)
        num_to_merge = int(T_intermediate * additional_merge_ratio / 2) * 2  # Make it even
        
        if num_to_merge > 4 and T_intermediate > 10:  # Only if we have enough tokens
            # Compute pairwise similarities for remaining tokens
            x_norm = x_intermediate / (x_intermediate.norm(dim=-1, keepdim=True) + 1e-6)
            sim_matrix = x_norm @ x_norm.transpose(-1, -2)  # [B, T, T]
            
            # Use bipartite matching: merge most similar pairs
            # Simple greedy approach: find top num_to_merge/2 pairs
            sim_matrix_flat = sim_matrix[0].clone()
            # Zero out diagonal
            sim_matrix_flat.fill_diagonal_(-float('inf'))
            
            merged_pairs = []
            used = set()
            
            for _ in range(num_to_merge // 2):
                # Find max similarity
                flat_idx = sim_matrix_flat.view(-1).argmax()
                i = flat_idx // T_intermediate
                j = flat_idx % T_intermediate
                i, j = i.item(), j.item()
                
                if i not in used and j not in used:
                    merged_pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    
                # Zero out this pair
                sim_matrix_flat[i, :] = -float('inf')
                sim_matrix_flat[:, i] = -float('inf')
                sim_matrix_flat[j, :] = -float('inf')
                sim_matrix_flat[:, j] = -float('inf')
            
            # Build final reduced sequence
            final_tokens = []
            skip_indices = set()
            
            for pair_idx, (i, j) in enumerate(merged_pairs):
                # Merge tokens i and j
                merged = (x_intermediate[:, i:i+1, :] + x_intermediate[:, j:j+1, :]) / 2
                final_tokens.append(merged)
                skip_indices.add(i)
                skip_indices.add(j)
            
            # Add unmerged tokens
            for idx in range(T_intermediate):
                if idx not in skip_indices:
                    final_tokens.append(x_intermediate[:, idx:idx+1, :])
            
            x = torch.cat(final_tokens, dim=1)
        else:
            x = x_intermediate
        
        T_new = x.shape[1]
        normed = block.norm1(x)
        
        # Compute Q, K, V for ALL tokens in reduced sequence
        qkv = block.attn.qkv(normed)
        qkv = qkv.reshape(B, T_new, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        tokens_kv_computed = T_new
        
        T = T_new  # Use reduced sequence length
    
    # Compute attention with (potentially reduced) sequence
    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    attn = attn.softmax(dim=-1)
    
    # Apply attention to values
    out = (attn @ v).transpose(1, 2).reshape(B, T, C)
    out = block.attn.proj(out)
    out = block.attn.proj_drop(out)
    
    # Residual
    x = x + block.drop_path1(out)
    
    # FFN
    x = x + block.drop_path2(block.mlp(block.norm2(x)))
    
    # Count FLOPs (with potentially reduced T)
    proj_flops = B * T * C * C * 4  # Q, K, V, Output projections
    attn_flops = count_attention_flops(B, num_heads, T, head_dim)
    
    flops = proj_flops + attn_flops
    
    return x, attn, k, v, flops

def build_cache_from_frames(model, cache_frames, num_saliency_layers=6, device='cuda', r_value=0.5):
    """
    Build K/V cache by iteratively merging background tokens across layers.
    
    NEW Strategy:
    - Process all cache frames through saliency layers normally
    - At last saliency layer: identify and iteratively merge BG tokens
    - Continue merging through remaining layers
    - Store only the FINAL layer's merged BG tokens in cache
    
    Args:
        num_saliency_layers: Number of layers to run normally (1-11). Remaining layers will be cached.
    
    Returns: cache_dict with final layer merged_bg_summary, K/V for cached layers, and overhead_flops
    """
    first_cached_layer = num_saliency_layers
    num_cached_layers = 12 - num_saliency_layers
    
    print(f"\nBuilding cache from {len(cache_frames)} frames with iterative merging...")
    print(f"  Saliency layers: 0-{num_saliency_layers-1} ({num_saliency_layers} layers)")
    print(f"  Cached layers: {first_cached_layer}-11 ({num_cached_layers} layers)")
    
    # Stack all cache frames
    cache_batch = torch.stack(cache_frames).to(device)
    B = cache_batch.shape[0]
    
    overhead_flops = 0  # Track overhead FLOPs
    
    with torch.no_grad():
        # Get patch embeddings
        x = model.patch_embed(cache_batch)
        cls_token = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, x), dim=1)
        tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)
        
        # Run saliency blocks normally on all frames
        print(f"  Processing {B} frames through blocks 0-{num_saliency_layers-1}...")
        for block_idx in range(num_saliency_layers):
            block = model.blocks[block_idx]
            tokens, _, _, _, _ = run_attention_block(block, tokens)
        
        # Use last saliency layer to identify and merge BG tokens
        B_curr, T, C = tokens.shape
        num_heads = model.blocks[0].attn.num_heads
        head_dim = C // num_heads
        
        # Get attention from last saliency block
        last_saliency_block = model.blocks[num_saliency_layers - 1]
        normed = last_saliency_block.norm1(tokens)
        qkv = last_saliency_block.attn.qkv(normed)
        qkv = qkv.reshape(B_curr, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        attn = attn.softmax(dim=-1)
        
        # Apply ConToMe with source tracking
        info = {
            "use": True,
            "r": [r_value] * 12,
            "size": None,
            "source_trace": True,  # Enable source tracking
            "source": None
        }
        
        print(f"\n  Merging BG tokens across {B_curr} frames using ConToMe...")
        fg_tokens, merged_bg_summary, k_merged, v_merged = ConToMe(
            tokens, attn, info, layer=num_saliency_layers-1, K=k, V=v
        )
        
        print(f"    Layer {num_saliency_layers-1}: BG tokens {tokens.shape[1] - fg_tokens.shape[1]} → {merged_bg_summary.shape[1]} (single merge pass)")
        print(f"  ✓ Merged BG summary: {merged_bg_summary.shape[1]} tokens")
        
        # Track merge statistics (source tracking may be in different formats)
        # Skip detailed statistics to avoid format issues
        if info.get('source') is not None:
            try:
                # Try to get some basic stats
                source = info['source']
                if isinstance(source, dict):
                    merge_counts = [len(v) if isinstance(v, (list, tuple)) else 1 for v in source.values()]
                    if merge_counts:
                        print(f"    Merge stats - min: {min(merge_counts)}, max: {max(merge_counts)}, mean: {sum(merge_counts)/len(merge_counts):.1f}")
                elif isinstance(source, (list, tuple)):
                    print(f"    Source tracking enabled ({len(source)} source entries)")
            except Exception as e:
                # If we can't parse source tracking, just note it's enabled
                print(f"    Source tracking enabled")
        
        # Estimate overhead for cache building (one-time cost)
        # ConToMe merging: bipartite matching and token averaging
        T_bg_before = tokens.shape[1] - fg_tokens.shape[1]
        overhead_flops += B_curr * T_bg_before * 100  # Approximate cost for ConToMe operations
        
        # Get source tracking info
        source_map = info.get('source', None)
        bg_tokens_original = info.get('bg_tokens_original', None)
        
        # Now process through cached layers using the merged BG representation
        tokens_for_cache = merged_bg_summary  # [1, T_bg_merged, C]
        
        cache_dict = {
            'bg_summary': merged_bg_summary,
            'source_map': source_map,
            'bg_tokens_original': bg_tokens_original,
            'overhead_flops': overhead_flops,
            'num_saliency_layers': num_saliency_layers
        }
        
        print(f"\n  Building K/V cache for layers {first_cached_layer}-11...")
        for layer_idx in range(first_cached_layer, 12):
            block = model.blocks[layer_idx]
            
            # Get K,V for the merged BG tokens
            normed = block.norm1(tokens_for_cache)
            qkv = block.attn.qkv(normed)
            B_c, T_c, _ = tokens_for_cache.shape
            qkv = qkv.reshape(B_c, T_c, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q_c, k_c, v_c = qkv[0], qkv[1], qkv[2]
            
            # Store K,V for this layer
            cache_dict[layer_idx] = {
                'K': k_c.clone(),
                'V': v_c.clone(),
            }
            
            # Continue processing for next layer
            tokens_for_cache, _, _, _, _ = run_attention_block(block, tokens_for_cache)
        
        print(f"\n  ✓ Cache built successfully:")
        print(f"    Final BG summary tokens: {merged_bg_summary.shape[1]}")
        print(f"    Layers cached: {first_cached_layer}-11 ({num_cached_layers} layers)")
        print(f"    Merging overhead: {overhead_flops/1e6:.2f}M FLOPs")
        
        return cache_dict

def run_with_kv_cache(model, test_img_tensor, cache_dict, threshold=0.5, device='cuda'):
    """
    Run ViT on test image using cached BG K/V
    
    Strategy:
    - Run saliency blocks normally to get test frame representation
    - Use last saliency layer attention to identify BG tokens in test frame
    - Match test BG tokens with cached BG summary using similarity
    - For cached layers: reuse cached K/V for matched BG tokens
    
    Returns: logits, total_flops, total_num_cached, total_overhead
    """
    B = test_img_tensor.shape[0]
    num_saliency_layers = cache_dict['num_saliency_layers']
    first_cached_layer = num_saliency_layers
    
    with torch.no_grad():
        # Get patch embeddings
        x = model.patch_embed(test_img_tensor)
        cls_token = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, x), dim=1)
        tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)
        
        total_flops = 0
        total_cached = 0
        num_heads = model.blocks[0].attn.num_heads
        _, T, C = tokens.shape
        head_dim = C // num_heads
        
        # Run saliency blocks normally (no caching)
        for block_idx in range(num_saliency_layers):
            block = model.blocks[block_idx]
            tokens, attn, k, v, flops = run_attention_block(block, tokens)
            total_flops += flops
            
            # Save last saliency layer's attention for BG detection
            if block_idx == num_saliency_layers - 1:
                attn_saliency = attn
        
        # Use last saliency layer attention to identify BG tokens in test frame
        score_obj = get_objective_score(attn_saliency)
        bg_mask = (score_obj.squeeze(-1) == 0)
        idx_bg = [torch.nonzero(bg_mask[b], as_tuple=False).squeeze(-1) for b in range(B)]
        
        # Extract BG tokens from test frame
        bg_tokens_per_batch = [tokens[b, idx, :] for b, idx in enumerate(idx_bg)]
        
        from torch.nn.utils.rnn import pad_sequence
        bg_tokens = pad_sequence(bg_tokens_per_batch, batch_first=True, padding_value=0.0)
        
        # Match test BG tokens with cached BG summary
        bg_summary = cache_dict['bg_summary']  # [1, T_bg_merged, C]
        
        # OVERHEAD: Token matching (PER TEST FRAME)
        # This is the cost of matching test BG tokens to cached BG summary
        # - Normalization: O(T_bg_test * C + T_bg_cached * C)
        # - Cosine similarity: O(T_bg_test * T_bg_cached * C)
        # - Finding max: O(T_bg_test * T_bg_cached)
        T_bg_test = bg_tokens.shape[1]
        T_bg_cached = bg_summary.shape[1]
        
        # Normalization (2 norm operations)
        matching_overhead = (T_bg_test + T_bg_cached) * C * 2  
        
        # Cosine similarity matrix computation: each test BG token compared to all cached BG
        matching_overhead += T_bg_test * T_bg_cached * C  
        
        # Finding best match (max operation) for each test token
        matching_overhead += T_bg_test * T_bg_cached  # Comparison operations
        
        a = bg_tokens / (bg_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        b = bg_summary / (bg_summary.norm(dim=-1, keepdim=True) + 1e-6)
        
        scores = (a @ b.transpose(-1, -2) + 1) / 2  # [B, T_bg_test, T_bg_cached]
        sim_val, sim_idx = scores.max(dim=-1)  # For each test BG token, find best cache match
        
        # DEBUG: Check similarity distribution
        if sim_val.numel() > 0:
            min_sim = sim_val.min().item()
            max_sim = sim_val.max().item()
            mean_sim = sim_val.mean().item()
            print(f"\n  📈 Similarity Distribution:")
            print(f"     Min: {min_sim:.3f}, Max: {max_sim:.3f}, Mean: {mean_sim:.3f}")
            print(f"     Tokens > {threshold:.2f}: {(sim_val[0] > threshold).sum().item()}/{sim_val.shape[1]}")
        
        # Find matched tokens (above threshold)
        matched_indices = []
        cache_kv_indices = []
        matched_similarities = []  # Track similarities
        
        for b_idx in range(B):
            matched = (sim_val[b_idx] > threshold).nonzero(as_tuple=False).squeeze(-1)
            if matched.numel() > 0:
                for m in matched:
                    if idx_bg[b_idx].numel() > m:
                        # This is the token index in the test frame
                        token_idx = idx_bg[b_idx][m].item()
                        # This is the corresponding index in the cached BG summary
                        cache_idx = sim_idx[b_idx, m].item()
                        similarity = sim_val[b_idx, m].item()
                        matched_indices.append(token_idx)
                        cache_kv_indices.append(cache_idx)
                        matched_similarities.append(similarity)
        
        num_bg_test = idx_bg[0].numel() if len(idx_bg) > 0 else 0
        num_bg_cached = bg_summary.shape[1]
        num_matched = len(matched_indices)
        unique_cache_indices = len(set(cache_kv_indices))
        
        # Get merging and matching overhead
        merging_overhead = cache_dict.get('overhead_flops', 0)
        
        print(f"\n  🔍 Matching Stats:")
        print(f"     Saliency layers: 0-{num_saliency_layers-1}")
        print(f"     Cached layers: {first_cached_layer}-11")
        print(f"     Test BG tokens: {num_bg_test}")
        print(f"     Cached BG tokens: {num_bg_cached}")
        print(f"     Matched test tokens: {num_matched} ({num_matched/num_bg_test*100 if num_bg_test > 0 else 0:.1f}% of test BG)")
        print(f"     Unique cache hits: {unique_cache_indices} ({unique_cache_indices/num_bg_cached*100 if num_bg_cached > 0 else 0:.1f}% of cache)")
        print(f"     Avg reuse per cached token: {num_matched/unique_cache_indices if unique_cache_indices > 0 else 0:.1f}x")
        print(f"     Threshold: {threshold:.2f}")
        print(f"\n  📊 Overhead FLOPs:")
        print(f"     Merging (cache build): {merging_overhead/1e6:.2f}M")
        print(f"     Matching (inference): {matching_overhead/1e6:.2f}M")
        print(f"     Total overhead: {(merging_overhead + matching_overhead)/1e6:.2f}M")
        num_cached_layers = 12 - num_saliency_layers
        print(f"     ℹ️  Note: 'Matched' in summary = {num_matched} × {num_cached_layers} layers = {num_matched * num_cached_layers} total")
        
        # Calculate similarity-based grouping
        # Use the threshold parameter to control merging aggressiveness
        # All tokens above threshold will be merged
        # We merge all matched test tokens that point to the same cache token
        
        merge_groups = {}  # cache_idx -> [(test_idx, similarity)] for all matched tokens
        
        for test_idx, cache_idx, similarity in zip(matched_indices, cache_kv_indices, matched_similarities):
            # All tokens that passed the threshold get merged
            if cache_idx not in merge_groups:
                merge_groups[cache_idx] = []
            merge_groups[cache_idx].append((test_idx, similarity))
        
        # Separate into high and medium confidence for better tracking
        high_conf_matches = {}  # sim > threshold + 0.2
        medium_conf_matches = {}  # sim > threshold
        
        high_threshold = min(threshold + 0.2, 0.95)  # Adaptive high threshold
        
        for cache_idx, token_list in merge_groups.items():
            high_conf = [t[0] for t in token_list if t[1] > high_threshold]
            medium_conf = [t[0] for t in token_list if t[1] <= high_threshold]
            
            if high_conf:
                high_conf_matches[cache_idx] = high_conf
            if medium_conf:
                medium_conf_matches[cache_idx] = medium_conf
        
        # Calculate sequence reduction
        num_high_conf = sum(len(indices) for indices in high_conf_matches.values())
        num_medium_conf = sum(len(indices) for indices in medium_conf_matches.values())
        
        tokens_after_high = num_high_conf - len(high_conf_matches) if high_conf_matches else 0
        tokens_after_medium = num_medium_conf - len(medium_conf_matches) if medium_conf_matches else 0
        total_reduction = tokens_after_high + tokens_after_medium
        
        if total_reduction > 0:
            sequence_reduction = total_reduction / T * 100
            print(f"     🎯 Sequence reduction strategy:")
            print(f"        High-conf (>{high_threshold:.2f}): {num_high_conf} tokens → {len(high_conf_matches)} groups = {tokens_after_high} removed")
            print(f"        Medium-conf (>{threshold:.2f}): {num_medium_conf} tokens → {len(medium_conf_matches)} groups = {tokens_after_medium} removed")
            print(f"        Total reduction: {total_reduction} tokens → {sequence_reduction:.1f}% shorter sequence")
        
        # For cached blocks: use THREE-TIER strategy
        for layer_idx in range(first_cached_layer, 12):
            block = model.blocks[layer_idx]
            layer_cache = cache_dict[layer_idx]
            
            # Prepare cache info with aggressive merging strategy
            if len(matched_indices) > 0:
                # Progressive merge ratio: more aggressive in later layers
                layer_progress = (layer_idx - first_cached_layer) / (12 - first_cached_layer)
                merge_ratio = 0.3 + 0.2 * layer_progress  # 0.3 → 0.5 as we go deeper
                
                cache_info = {
                    'matched_indices': matched_indices,
                    'cache_kv_indices': cache_kv_indices,
                    'cached_K': layer_cache['K'],
                    'cached_V': layer_cache['V'],
                    'bg_summary': bg_summary,
                    'high_confidence_matches': high_conf_matches,
                    'medium_confidence_matches': medium_conf_matches,
                    'merge_ratio': merge_ratio,  # Progressive merging
                }
                
                total_cached += len(matched_indices)
            else:
                cache_info = None
            
            # Run block with progressive merging
            tokens, _, _, _, flops = run_attention_block(block, tokens, cache_info=cache_info)
            total_flops += flops
        
        # Final classification
        tokens = model.norm(tokens)
        logits = model.head(tokens[:, 0])
        
    total_overhead = merging_overhead + matching_overhead
    
    # IMPORTANT: merging_overhead is CACHE BUILDING cost (one-time)
    # matching_overhead is PER-FRAME cost
    # Return them separately so they can be properly amortized
    return logits, total_flops, total_cached, matching_overhead, merging_overhead

def run_baseline_vit(model, img_tensor, device='cuda'):
    """Run baseline ViT without any caching"""
    B = img_tensor.shape[0]
    num_heads = model.blocks[0].attn.num_heads
    
    with torch.no_grad():
        x = model.patch_embed(img_tensor)
        cls_token = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, x), dim=1)
        tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)
        
        _, T, C = tokens.shape
        head_dim = C // num_heads
        
        # Run all 12 blocks
        total_flops = 0
        for block_idx in range(12):
            block = model.blocks[block_idx]
            tokens, _, _, _, flops = run_attention_block(block, tokens)
            total_flops += flops
        
        # Classification
        tokens = model.norm(tokens)
        logits = model.head(tokens[:, 0])
        
    return logits, total_flops

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading ViT model...")
    model = timm.create_model(
        'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        pretrained=True,
        num_classes=1000
    ).to(device)
    model.eval()
    print("Model loaded\n")
    
    # Load class names
    print("Loading ImageNet class names...")
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        class_names = requests.get(url, timeout=10).json()
    except:
        class_names = [f"class_{i}" for i in range(1000)]
    
    # Load images
    data_config = timm.data.resolve_model_data_config(model)
    transforms_val = timm.data.create_transform(**data_config, is_training=False)
    
    frames_dir = "goldfish_imgs"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    # Split frames in half
    total_frames = len(frame_files)
    split_point = total_frames // 2
    
    print(f"Found {total_frames} frames")
    print(f"Using first {split_point} frames for cache, remaining {total_frames - split_point} for testing\n")
    
    # Load all frames
    all_imgs = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in frame_files]
    all_tensors = [transforms_val(img) for img in all_imgs]
    
    # Split into cache and test sets
    cache_tensors = all_tensors[:split_point]  # First half
    test_tensors = all_tensors[split_point:]  # Second half
    test_frame_names = frame_files[split_point:]
    
    # ========================================================================
    # Test 1: Baseline ViT on Test Set
    # ========================================================================
    print("="*80)
    print("Test 1: Baseline ViT (No Caching) on Test Set")
    print("="*80)
    print("Running all 12 blocks normally on test frames...")
    
    baseline_results = []
    baseline_top1_preds = []  # Store top-1 predictions for accuracy
    baseline_top5_preds = []  # Store top-5 predictions for accuracy
    baseline_ground_truths = []  # Store ground truth labels
    total_baseline_flops = 0
    
    # ASSUMPTION: Since we're using video frames, we assume they all belong to the same class
    # We'll extract ground truth from the most common baseline prediction
    
    for idx, test_tensor in enumerate(test_tensors):
        test_tensor_batch = test_tensor.unsqueeze(0).to(device)
        logits_baseline, flops_baseline = run_baseline_vit(model, test_tensor_batch, device)
        probs_baseline = torch.softmax(logits_baseline.cpu(), dim=-1)
        conf_baseline, pred_baseline = torch.max(probs_baseline, dim=-1)
        
        # Get top-5 predictions
        top5_indices = get_top_k_predictions(logits_baseline.cpu(), k=5)[0]
        
        baseline_results.append({
            'frame_name': test_frame_names[idx],
            'prediction': class_names[pred_baseline.item()],
            'pred_idx': pred_baseline.item(),
            'confidence': conf_baseline.item(),
            'flops': flops_baseline,
            'top5_indices': top5_indices
        })
        baseline_top1_preds.append(pred_baseline.item())
        baseline_top5_preds.append(top5_indices)
        total_baseline_flops += flops_baseline
        
        print(f"\n  Frame {idx+1}/{len(test_tensors)}: {test_frame_names[idx]}")
        print(f"    Prediction: {class_names[pred_baseline.item()]}")
        print(f"    Confidence: {conf_baseline.item():.4f}")
        print(f"    FLOPs: {flops_baseline/1e6:.2f}M")
    
    # Determine ground truth: use the most common baseline prediction
    # (Assumption: video frames should mostly classify to same object)
    from collections import Counter
    pred_counter = Counter(baseline_top1_preds)
    most_common_pred, count = pred_counter.most_common(1)[0]
    ground_truth_class = most_common_pred
    ground_truth_name = class_names[ground_truth_class]
    baseline_ground_truths = [ground_truth_class] * len(test_tensors)
    
    # Calculate baseline Top-1 and Top-5 accuracy
    baseline_top1_acc = calculate_top_k_accuracy(baseline_top1_preds, baseline_ground_truths, k=1)
    baseline_top5_acc = calculate_top_k_accuracy(baseline_top5_preds, baseline_ground_truths, k=5)
    
    avg_baseline_flops = total_baseline_flops / len(test_tensors)
    avg_baseline_conf = sum(r['confidence'] for r in baseline_results) / len(baseline_results)
    
    print(f"\n  Average Baseline FLOPs: {avg_baseline_flops/1e6:.2f}M")
    print(f"  Average Baseline Confidence: {avg_baseline_conf:.4f}")
    print(f"  Total Baseline FLOPs: {total_baseline_flops/1e6:.2f}M")
    
    print(f"\n  📊 BASELINE ACCURACY:")
    print(f"     Ground Truth Class: {ground_truth_name} (appears in {count}/{len(test_tensors)} frames)")
    print(f"     Top-1 Accuracy: {baseline_top1_acc*100:.2f}%")
    print(f"     Top-5 Accuracy: {baseline_top5_acc*100:.2f}%")
    
    # Show what labels are being classified
    print(f"\n  🏷️  Classification Summary:")
    unique_predictions = set(r['prediction'] for r in baseline_results)
    for pred_class in sorted(unique_predictions):
        count = sum(1 for r in baseline_results if r['prediction'] == pred_class)
        frames = [r['frame_name'] for r in baseline_results if r['prediction'] == pred_class]
        print(f"    • {pred_class}: {count} frame(s) - {', '.join(frames)}")
    
    # ========================================================================
    # Test Multiple Saliency Layer Configurations, R Values and Thresholds
    # ========================================================================
    print("\n" + "="*80)
    print("Testing K/V Caching with Multiple Configurations")
    print("="*80)
    
    # Configuration space
    # STRATEGY: Test broader range of hyperparameters
    # Since BG similarities are 0.76-1.0, threshold mainly acts as on/off switch
    # Real control comes from how aggressively we merge within matched tokens
    saliency_layer_configs = [3, 4, 5, 6]  # Test more saliency configurations
    r_values = [0.3, 0.5, 0.7]  # R values for ConToMe merging (currently not used effectively)
    thresholds = [0.60, 0.70, 0.75, 0.80, 0.85]  # Broader range of thresholds
    
    print(f"\nSaliency layer configs: {saliency_layer_configs}")
    print(f"R values: {r_values} (note: progressive merge_ratio used in cached layers)")
    print(f"Thresholds: {thresholds}")
    print(f"Total configurations: {len(saliency_layer_configs) * len(r_values) * len(thresholds)}")
    
    all_results = []
    
    for num_sal in saliency_layer_configs:
        for r_val in r_values:
            print(f"\n{'='*80}")
            print(f"Testing: {num_sal} saliency layers, r = {r_val:.1f}")
            print(f"{'='*80}")
            
            # Build cache with this configuration (using first half of frames)
            cache_dict = build_cache_from_frames(model, cache_tensors, num_sal, device, r_value=r_val)
            
            # Test each threshold with this cache on all test frames
            for threshold in thresholds:
                print(f"\n{'='*40}")
                print(f"  Saliency={num_sal}, r={r_val:.1f}, Threshold={threshold:.2f}")
                print(f"{'='*40}")
                
                total_flops_all_frames = 0
                total_cached_all_frames = 0
                total_matching_overhead_all_frames = 0  # Per-frame overhead
                cache_build_overhead = 0  # One-time overhead (will be amortized)
                test_frame_results = []
                cached_top1_preds = []
                cached_top5_preds = []
                
                # Test on all frames in test set
                for idx, test_tensor in enumerate(test_tensors):
                    test_tensor_batch = test_tensor.unsqueeze(0).to(device)
                    
                    logits, flops, cached, matching_overhead, merging_overhead = run_with_kv_cache(
                        model, test_tensor_batch, cache_dict, threshold=threshold, device=device
                    )
                    probs = torch.softmax(logits.cpu(), dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    
                    # Get top-5 predictions
                    top5_indices = get_top_k_predictions(logits.cpu(), k=5)[0]
                    
                    total_flops_all_frames += flops
                    total_cached_all_frames += cached
                    total_matching_overhead_all_frames += matching_overhead
                    
                    # Cache building overhead is same for all frames (one-time cost)
                    if idx == 0:
                        cache_build_overhead = merging_overhead
                    
                    test_frame_results.append({
                        'frame_name': test_frame_names[idx],
                        'prediction': class_names[pred.item()],
                        'pred_idx': pred.item(),
                        'confidence': conf.item(),
                        'flops': flops,
                        'cached': cached,
                        'matching_overhead': matching_overhead,
                        'top5_indices': top5_indices
                    })
                    cached_top1_preds.append(pred.item())
                    cached_top5_preds.append(top5_indices)
                
                # Calculate Top-1 and Top-5 accuracy for cached inference
                cached_top1_acc = calculate_top_k_accuracy(cached_top1_preds, baseline_ground_truths, k=1)
                cached_top5_acc = calculate_top_k_accuracy(cached_top5_preds, baseline_ground_truths, k=5)
                
                # Calculate Top-1 and Top-5 accuracy for cached inference
                cached_top1_acc = calculate_top_k_accuracy(cached_top1_preds, baseline_ground_truths, k=1)
                cached_top5_acc = calculate_top_k_accuracy(cached_top5_preds, baseline_ground_truths, k=5)
                
                # Average metrics across test set
                avg_flops = total_flops_all_frames / len(test_tensors)
                avg_cached = total_cached_all_frames / len(test_tensors)
                avg_matching_overhead = total_matching_overhead_all_frames / len(test_tensors)
                
                # Amortize cache building overhead across all test frames
                avg_cache_build_overhead = cache_build_overhead / len(test_tensors)
                avg_total_overhead = avg_matching_overhead + avg_cache_build_overhead
                avg_confidence = sum(r['confidence'] for r in test_frame_results) / len(test_frame_results)
                
                # Calculate reduction compared to average baseline
                baseline_ref = baseline_results[0]['flops']  # Use first frame baseline as reference
                
                result = {
                    'num_saliency_layers': num_sal,
                    'num_cached_layers': 12 - num_sal,
                    'r_value': r_val,
                    'threshold': threshold,
                    'avg_flops': avg_flops,
                    'avg_cached': avg_cached,
                    'avg_overhead': avg_total_overhead,  # Total overhead (matching + amortized cache build)
                    'avg_matching_overhead': avg_matching_overhead,  # Per-frame matching
                    'avg_cache_build_overhead': avg_cache_build_overhead,  # Amortized cache building
                    'avg_confidence': avg_confidence,
                    'top1_accuracy': cached_top1_acc,
                    'top5_accuracy': cached_top5_acc,
                    'test_frame_results': test_frame_results,
                    'bg_summary_tokens': cache_dict['bg_summary'].shape[1],
                    'num_test_frames': len(test_tensors)
                }
                all_results.append(result)
                
                net_flops = avg_flops + avg_total_overhead
                savings = baseline_ref - avg_flops
                net_savings = baseline_ref - net_flops
                
                print(f"\n  Test Set Summary ({len(test_tensors)} frames):")
                print(f"    Avg Attention FLOPs: {avg_flops/1e6:.2f}M")
                print(f"    Avg Matching Overhead: {avg_matching_overhead/1e6:.2f}M (per frame)")
                print(f"    Avg Cache Build Overhead: {avg_cache_build_overhead/1e6:.2f}M (amortized over {len(test_tensors)} frames)")
                print(f"    Avg Total Overhead: {avg_total_overhead/1e6:.2f}M")
                print(f"    Avg Net FLOPs: {net_flops/1e6:.2f}M")
                print(f"    Avg Tokens cached: {avg_cached:.1f}")
                print(f"    Avg Confidence: {avg_confidence:.4f}")
                print(f"    Gross FLOP reduction: {(savings/baseline_ref)*100:.2f}%")
                print(f"    Net FLOP reduction (after overhead): {(net_savings/baseline_ref)*100:.2f}%")
                print(f"\n    📊 ACCURACY:")
                print(f"       Top-1 Accuracy: {cached_top1_acc*100:.2f}% (Baseline: {baseline_top1_acc*100:.2f}%)")
                print(f"       Top-5 Accuracy: {cached_top5_acc*100:.2f}% (Baseline: {baseline_top5_acc*100:.2f}%)")
                print(f"       Top-1 Drop: {(baseline_top1_acc - cached_top1_acc)*100:+.2f}%")
                print(f"       Top-5 Drop: {(baseline_top5_acc - cached_top5_acc)*100:+.2f}%")
                
                # Show predicted labels for this configuration
                print(f"\n    🏷️  Predictions:")
                for frame_res in test_frame_results:
                    print(f"      {frame_res['frame_name']}: {frame_res['prediction']} (conf: {frame_res['confidence']:.4f})")
    
    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "="*80)
    print("📊 UNDERSTANDING NET SAVINGS")
    print("="*80)
    print(f"""
The "Net Savings" metric shows the REAL computational savings after accounting for overhead:

• Baseline FLOPs: {baseline_results[0]['flops']/1e6:.2f}M 
  (Normal ViT running all 12 layers on full 197-token sequence)

• Cached Attention FLOPs: Reduced by reusing K/V cache from previous frames
  (Only compute attention for smaller merged sequence)

• Overhead FLOPs: Cost of merging tokens and matching to cache
  - Cache building: Token merging using ConToMe (~187M FLOPs)
  - Inference matching: Finding similar tokens between test and cache frames (~27-30M FLOPs)
  - Total overhead: ~214-446M FLOPs depending on configuration

• Net FLOPs = Cached Attention FLOPs + Overhead FLOPs

• Net Savings % = (Baseline FLOPs - Net FLOPs) / Baseline FLOPs × 100

Example: Best configuration (3 saliency, 9 cached layers, threshold 0.60):
  Baseline: {baseline_results[0]['flops']/1e6:.2f}M → Net: ~3746M = {((baseline_results[0]['flops'] - 3746e6)/baseline_results[0]['flops'])*100:.2f}% net savings

The overhead is amortized across video frames - it's a one-time cost for cache building!
""")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS - ALL CONFIGURATIONS (AVERAGED OVER TEST SET)")
    print("="*80)
    
    print("\n{:<8} {:<8} {:<6} {:<9} {:<12} {:<10} {:<10} {:<10} {:<12} {:<10} {:<10} {:<10}".format(
        "Saliency", "Cached", "r", "Thresh", "NetFLOPs(M)", "NetSave%", "Conf", "Top1%", "Top5%", "T1Drop%", "T5Drop%", "BGCache"
    ))
    print("-" * 145)
    
    baseline_ref = baseline_results[0]['flops']
    
    for res in all_results:
        net_reduction = (1 - (res['avg_flops'] + res['avg_overhead'])/baseline_ref) * 100
        net_flops_m = (res['avg_flops'] + res['avg_overhead'])/1e6
        top1_drop = (baseline_top1_acc - res['top1_accuracy'])*100
        top5_drop = (baseline_top5_acc - res['top5_accuracy'])*100
        print("{:<8} {:<8} {:<6.1f} {:<9.2f} {:<12.1f} {:<10.1f} {:<10.4f} {:<10.1f} {:<10.1f} {:<10.2f} {:<10.2f} {:<10}".format(
            res['num_saliency_layers'],
            res['num_cached_layers'],
            res['r_value'],
            res['threshold'],
            net_flops_m,
            net_reduction,
            res['avg_confidence'],
            res['top1_accuracy']*100,
            res['top5_accuracy']*100,
            top1_drop,
            top5_drop,
            res['bg_summary_tokens']
        ))
    
    # ========================================================================
    # Generate 2 Comparison Plots
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    # Calculate baseline averages
    avg_baseline_conf = sum(r['confidence'] for r in baseline_results) / len(baseline_results)
    avg_baseline_flops = sum(r['flops'] for r in baseline_results) / len(baseline_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Prepare data for all configurations
    config_labels = []
    config_confs = []
    config_baseline_flops = []
    config_cached_flops = []
    config_overheads = []
    
    for res in all_results:
        label = f"S{res['num_saliency_layers']}C{res['num_cached_layers']}\nr={res['r_value']:.1f}\nt={res['threshold']:.2f}"
        config_labels.append(label)
        config_confs.append(res['avg_confidence'])
        config_baseline_flops.append(avg_baseline_flops / 1e6)  # Convert to M FLOPs
        config_cached_flops.append(res['avg_flops'] / 1e6)
        config_overheads.append(res['avg_overhead'] / 1e6)
    
    x_pos = np.arange(len(config_labels))
    
    # Plot 1: Confidence Comparison
    ax1.bar(x_pos, [avg_baseline_conf] * len(config_labels), 
            width=0.8, alpha=0.3, color='red', label='Baseline', edgecolor='red', linewidth=2)
    ax1.bar(x_pos, config_confs, width=0.8, alpha=0.7, 
            color='blue', label='K/V Caching Configs', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Configuration (S=Saliency layers, C=Cached layers)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Confidence', fontweight='bold', fontsize=12)
    ax1.set_title('Confidence: Baseline vs All Configurations', fontweight='bold', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_labels, rotation=90, fontsize=7, ha='center')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add percentage difference labels on top of bars (select every 5th for readability)
    for i in range(0, len(config_confs), max(1, len(config_confs)//20)):
        conf = config_confs[i]
        baseline = avg_baseline_conf
        diff_pct = ((conf - baseline) / baseline) * 100
        color = 'green' if diff_pct > 0 else 'darkred'
        ax1.text(i, conf + 0.01, f'+{diff_pct:.1f}%' if diff_pct > 0 else f'{diff_pct:.1f}%', 
                ha='center', va='bottom', fontsize=6, fontweight='bold', color=color, rotation=0)
    
    # Plot 2: FLOP Comparison with Overhead
    bar_width = 0.25
    x_pos_flops = np.arange(len(config_labels))
    
    # Baseline FLOPs
    bars1 = ax2.bar(x_pos_flops - bar_width, config_baseline_flops, 
                    bar_width, label='Baseline FLOPs', 
                    color='red', alpha=0.6, edgecolor='black', linewidth=1)
    
    # Cached FLOPs (without overhead)
    bars2 = ax2.bar(x_pos_flops, config_cached_flops, 
                    bar_width, label='Cached FLOPs (gross)', 
                    color='blue', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Total FLOPs (cached + overhead)
    total_flops = [cached + overhead for cached, overhead in zip(config_cached_flops, config_overheads)]
    bars3 = ax2.bar(x_pos_flops + bar_width, total_flops, 
                    bar_width, label='Cached + Overhead (net)', 
                    color='orange', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add overhead stacked visualization
    bars4 = ax2.bar(x_pos_flops, config_overheads, 
                    bar_width, bottom=config_cached_flops,
                    color='darkred', alpha=0.5, edgecolor='black', linewidth=1, 
                    label='Overhead (stacked)', hatch='///')
    
    ax2.set_xlabel('Configuration (S=Saliency layers, C=Cached layers)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('FLOPs (Millions)', fontweight='bold', fontsize=12)
    ax2.set_title('FLOP Comparison: Baseline vs Cached vs Overhead', fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos_flops)
    ax2.set_xticklabels(config_labels, rotation=90, fontsize=7, ha='center')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add net FLOP reduction percentage labels (select every 5th for readability)
    for i in range(0, len(config_baseline_flops), max(1, len(config_baseline_flops)//20)):
        baseline = config_baseline_flops[i]
        total = total_flops[i]
        reduction = ((baseline - total) / baseline) * 100
        color = 'green' if reduction > 0 else 'darkred'
        y_pos = max(baseline, total) + 100
        ax2.text(i + bar_width, y_pos, f'{reduction:+.1f}%', 
                ha='center', va='bottom', fontsize=7, fontweight='bold', 
                color=color, rotation=0)
    
    plt.suptitle(f'K/V Caching Configuration Comparison\nBaseline: {avg_baseline_flops/1e6:.1f}M FLOPs, {avg_baseline_conf:.4f} Confidence\n({len(test_tensors)} test frames)',
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = 'vit_kv_caching_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved main comparison plots to: {output_file}")
    
    output_pdf = 'vit_kv_caching_comparison.pdf'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf}")
    
    # ========================================================================
    # Additional Analysis: R-value and Threshold Effects
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING R-VALUE AND THRESHOLD ANALYSIS PLOTS")
    print("="*80)
    
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get unique values
    saliency_vals = sorted(list(set([r['num_saliency_layers'] for r in all_results])))
    r_vals = sorted(list(set([r['r_value'] for r in all_results])))
    thresh_vals = sorted(list(set([r['threshold'] for r in all_results])))
    
    # Plot 1: Net FLOPs vs R-value (for each saliency config, average over thresholds)
    ax1 = axes[0, 0]
    for num_sal in saliency_vals:
        sal_data = []
        for r in r_vals:
            configs = [res for res in all_results 
                      if res['num_saliency_layers'] == num_sal and abs(res['r_value'] - r) < 0.01]
            if configs:
                avg_net_flops = np.mean([(c['avg_flops'] + c['avg_overhead'])/1e6 for c in configs])
                sal_data.append(avg_net_flops)
            else:
                sal_data.append(0)
        ax1.plot(r_vals, sal_data, 'o-', linewidth=2.5, markersize=10, 
                label=f'{num_sal} sal, {12-num_sal} cached', alpha=0.8)
    ax1.axhline(y=avg_baseline_flops/1e6, color='red', linestyle='--', linewidth=3, 
                alpha=0.7, label='Baseline')
    ax1.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Net FLOPs (M)', fontweight='bold', fontsize=13)
    ax1.set_title('Net FLOPs vs R Value\n(averaged over thresholds)', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Plot 2: Net FLOPs vs Threshold (for each saliency config, average over r values)
    ax2 = axes[0, 1]
    for num_sal in saliency_vals:
        sal_data = []
        for t in thresh_vals:
            configs = [res for res in all_results 
                      if res['num_saliency_layers'] == num_sal and abs(res['threshold'] - t) < 0.01]
            if configs:
                avg_net_flops = np.mean([(c['avg_flops'] + c['avg_overhead'])/1e6 for c in configs])
                sal_data.append(avg_net_flops)
            else:
                sal_data.append(0)
        ax2.plot(thresh_vals, sal_data, 's-', linewidth=2.5, markersize=10, 
                label=f'{num_sal} sal, {12-num_sal} cached', alpha=0.8)
    ax2.axhline(y=avg_baseline_flops/1e6, color='red', linestyle='--', linewidth=3, 
                alpha=0.7, label='Baseline')
    ax2.set_xlabel('Threshold', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Net FLOPs (M)', fontweight='bold', fontsize=13)
    ax2.set_title('Net FLOPs vs Threshold\n(averaged over r values)', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Plot 3: Confidence vs R-value
    ax3 = axes[1, 0]
    for num_sal in saliency_vals:
        sal_data = []
        for r in r_vals:
            configs = [res for res in all_results 
                      if res['num_saliency_layers'] == num_sal and abs(res['r_value'] - r) < 0.01]
            if configs:
                avg_conf = np.mean([c['avg_confidence'] for c in configs])
                sal_data.append(avg_conf)
            else:
                sal_data.append(0)
        ax3.plot(r_vals, sal_data, '^-', linewidth=2.5, markersize=10, 
                label=f'{num_sal} sal, {12-num_sal} cached', alpha=0.8)
    ax3.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=3, 
                alpha=0.7, label='Baseline')
    ax3.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Confidence', fontweight='bold', fontsize=13)
    ax3.set_title('Confidence vs R Value\n(averaged over thresholds)', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(alpha=0.3, linestyle='--')
    
    # Plot 4: Confidence vs Threshold
    ax4 = axes[1, 1]
    for num_sal in saliency_vals:
        sal_data = []
        for t in thresh_vals:
            configs = [res for res in all_results 
                      if res['num_saliency_layers'] == num_sal and abs(res['threshold'] - t) < 0.01]
            if configs:
                avg_conf = np.mean([c['avg_confidence'] for c in configs])
                sal_data.append(avg_conf)
            else:
                sal_data.append(0)
        ax4.plot(thresh_vals, sal_data, 'd-', linewidth=2.5, markersize=10, 
                label=f'{num_sal} sal, {12-num_sal} cached', alpha=0.8)
    ax4.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=3, 
                alpha=0.7, label='Baseline')
    ax4.set_xlabel('Threshold', fontweight='bold', fontsize=13)
    ax4.set_ylabel('Confidence', fontweight='bold', fontsize=13)
    ax4.set_title('Confidence vs Threshold\n(averaged over r values)', fontweight='bold', fontsize=14)
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle(f'R-Value and Threshold Analysis\nBaseline: {avg_baseline_flops/1e6:.1f}M FLOPs, {avg_baseline_conf:.4f} Confidence',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_file2 = 'vit_kv_caching_r_threshold_analysis.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved r-value and threshold analysis to: {output_file2}")
    
    output_pdf2 = 'vit_kv_caching_r_threshold_analysis.pdf'
    plt.savefig(output_pdf2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf2}")
    
    # ========================================================================
    # Best Configurations Plot
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING BEST CONFIGURATIONS PLOT")
    print("="*80)
    
    fig3, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Find best configuration for each saliency layer count
    best_configs = []
    for num_sal in saliency_vals:
        sal_results = [res for res in all_results if res['num_saliency_layers'] == num_sal]
        # Best = highest net FLOP reduction while maintaining confidence >= baseline
        valid_configs = [r for r in sal_results if r['avg_confidence'] >= avg_baseline_conf]
        if valid_configs:
            best = max(valid_configs, key=lambda x: (baseline_ref - (x['avg_flops'] + x['avg_overhead']))/baseline_ref)
        else:
            # If no config beats baseline confidence, just pick best FLOP reduction
            best = max(sal_results, key=lambda x: (baseline_ref - (x['avg_flops'] + x['avg_overhead']))/baseline_ref)
        best_configs.append(best)
    
    # Create grouped bar chart
    x_pos = np.arange(len(best_configs))
    bar_width = 0.25
    
    # Baseline FLOPs
    baseline_bars = ax.bar(x_pos - bar_width, [avg_baseline_flops/1e6] * len(best_configs),
                          bar_width, label='Baseline FLOPs', color='red', alpha=0.6, 
                          edgecolor='black', linewidth=2)
    
    # Net FLOPs
    net_flops_vals = [(c['avg_flops'] + c['avg_overhead'])/1e6 for c in best_configs]
    net_bars = ax.bar(x_pos, net_flops_vals, bar_width, label='Net FLOPs (Cached + Overhead)',
                     color='orange', alpha=0.7, edgecolor='black', linewidth=2)
    
    # Confidence (on secondary y-axis)
    ax2 = ax.twinx()
    conf_vals = [c['avg_confidence'] for c in best_configs]
    conf_bars = ax2.bar(x_pos + bar_width, conf_vals, bar_width, label='Confidence',
                       color='blue', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.axhline(y=avg_baseline_conf, color='blue', linestyle='--', linewidth=2.5, 
                alpha=0.7, label='Baseline Confidence')
    
    # Labels and formatting
    ax.set_xlabel('Layer Configuration', fontweight='bold', fontsize=13)
    ax.set_ylabel('FLOPs (Millions)', fontweight='bold', fontsize=13, color='black')
    ax2.set_ylabel('Confidence', fontweight='bold', fontsize=13, color='blue')
    ax.set_title('Best Configuration per Saliency Layer Count\n(maximizing FLOP reduction while maintaining confidence)',
                fontweight='bold', fontsize=14)
    
    config_labels = [f'{c["num_saliency_layers"]} sal\n{c["num_cached_layers"]} cached\nr={c["r_value"]:.1f}, t={c["threshold"]:.2f}' 
                    for c in best_configs]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, fontsize=10)
    
    # Add value labels on bars
    for i, (net_val, conf_val) in enumerate(zip(net_flops_vals, conf_vals)):
        reduction = ((avg_baseline_flops/1e6 - net_val) / (avg_baseline_flops/1e6)) * 100
        ax.text(i, net_val + 50, f'{reduction:+.1f}%', ha='center', va='bottom', 
               fontsize=9, fontweight='bold', color='darkgreen' if reduction > 0 else 'darkred')
        ax2.text(i + bar_width, conf_val + 0.01, f'{conf_val:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='darkblue')
    
    ax.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    output_file3 = 'vit_kv_caching_best_configs.png'
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"✓ Saved best configurations plot to: {output_file3}")
    
    output_pdf3 = 'vit_kv_caching_best_configs.pdf'
    plt.savefig(output_pdf3, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf3}")
    
    # ========================================================================
    # Comprehensive Hyperparameter Analysis (2x3 Grid)
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE HYPERPARAMETER ANALYSIS (2x3 GRID)")
    print("="*80)
    
    fig4, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Get unique values
    unique_saliency = sorted(list(set([r['num_saliency_layers'] for r in all_results])))
    unique_r = sorted(list(set([r['r_value'] for r in all_results])))
    unique_thresh = sorted(list(set([r['threshold'] for r in all_results])))
    
    # ========= ROW 1: NET FLOPs (with overhead) vs Parameters =========
    
    # Plot 1: Net FLOPs vs Saliency Layers (average over r and threshold)
    ax = axes[0, 0]
    saliency_net_flops = []
    saliency_net_flops_std = []
    for sal in unique_saliency:
        configs = [r for r in all_results if r['num_saliency_layers'] == sal]
        net_flops_vals = [(c['avg_flops'] + c['avg_overhead'])/1e6 for c in configs]
        saliency_net_flops.append(np.mean(net_flops_vals))
        saliency_net_flops_std.append(np.std(net_flops_vals))
    
    ax.errorbar(unique_saliency, saliency_net_flops, yerr=saliency_net_flops_std,
                marker='o', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkblue', ecolor='steelblue', label='Net FLOPs (Cached + Overhead)')
    ax.axhline(y=avg_baseline_flops/1e6, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline FLOPs')
    ax.set_xlabel('Number of Saliency Layers', fontweight='bold', fontsize=13)
    ax.set_ylabel('Net FLOPs (Millions)', fontweight='bold', fontsize=13)
    ax.set_title('Net FLOPs vs Saliency Layers\n(averaged over r and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (sal, net_flop) in enumerate(zip(unique_saliency, saliency_net_flops)):
        reduction = ((avg_baseline_flops/1e6 - net_flop) / (avg_baseline_flops/1e6)) * 100
        color = 'green' if reduction > 0 else 'red'
        ax.text(sal, net_flop + 100, f'{reduction:+.1f}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    # Plot 2: Net FLOPs vs R Value (average over saliency and threshold)
    ax = axes[0, 1]
    r_net_flops = []
    r_net_flops_std = []
    for r in unique_r:
        configs = [res for res in all_results if abs(res['r_value'] - r) < 0.01]
        net_flops_vals = [(c['avg_flops'] + c['avg_overhead'])/1e6 for c in configs]
        r_net_flops.append(np.mean(net_flops_vals))
        r_net_flops_std.append(np.std(net_flops_vals))
    
    ax.errorbar(unique_r, r_net_flops, yerr=r_net_flops_std,
                marker='s', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkgreen', ecolor='lightgreen', label='Net FLOPs (Cached + Overhead)')
    ax.axhline(y=avg_baseline_flops/1e6, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline FLOPs')
    ax.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax.set_ylabel('Net FLOPs (Millions)', fontweight='bold', fontsize=13)
    ax.set_title('Net FLOPs vs R Value\n(averaged over saliency and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (r, net_flop) in enumerate(zip(unique_r, r_net_flops)):
        reduction = ((avg_baseline_flops/1e6 - net_flop) / (avg_baseline_flops/1e6)) * 100
        color = 'green' if reduction > 0 else 'red'
        ax.text(r, net_flop + 100, f'{reduction:+.1f}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    # Plot 3: Net FLOPs vs Threshold (average over saliency and r)
    ax = axes[0, 2]
    thresh_net_flops = []
    thresh_net_flops_std = []
    for t in unique_thresh:
        configs = [res for res in all_results if abs(res['threshold'] - t) < 0.01]
        net_flops_vals = [(c['avg_flops'] + c['avg_overhead'])/1e6 for c in configs]
        thresh_net_flops.append(np.mean(net_flops_vals))
        thresh_net_flops_std.append(np.std(net_flops_vals))
    
    ax.errorbar(unique_thresh, thresh_net_flops, yerr=thresh_net_flops_std,
                marker='^', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkorange', ecolor='orange', label='Net FLOPs (Cached + Overhead)')
    ax.axhline(y=avg_baseline_flops/1e6, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline FLOPs')
    ax.set_xlabel('Matching Threshold', fontweight='bold', fontsize=13)
    ax.set_ylabel('Net FLOPs (Millions)', fontweight='bold', fontsize=13)
    ax.set_title('Net FLOPs vs Threshold\n(averaged over saliency and r)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (t, net_flop) in enumerate(zip(unique_thresh, thresh_net_flops)):
        reduction = ((avg_baseline_flops/1e6 - net_flop) / (avg_baseline_flops/1e6)) * 100
        color = 'green' if reduction > 0 else 'red'
        ax.text(t, net_flop + 100, f'{reduction:+.1f}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    # ========= ROW 2: CONFIDENCE vs Parameters =========
    
    # Plot 4: Confidence vs Saliency Layers
    ax = axes[1, 0]
    saliency_conf = []
    saliency_conf_std = []
    for sal in unique_saliency:
        configs = [r for r in all_results if r['num_saliency_layers'] == sal]
        conf_vals = [c['avg_confidence'] for c in configs]
        saliency_conf.append(np.mean(conf_vals))
        saliency_conf_std.append(np.std(conf_vals))
    
    ax.errorbar(unique_saliency, saliency_conf, yerr=saliency_conf_std,
                marker='o', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkblue', ecolor='steelblue', label='Confidence')
    ax.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Confidence')
    ax.set_xlabel('Number of Saliency Layers', fontweight='bold', fontsize=13)
    ax.set_ylabel('Confidence', fontweight='bold', fontsize=13)
    ax.set_title('Confidence vs Saliency Layers\n(averaged over r and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add confidence delta labels
    for i, (sal, conf) in enumerate(zip(unique_saliency, saliency_conf)):
        delta = conf - avg_baseline_conf
        color = 'green' if delta >= -0.01 else 'red'
        ax.text(sal, conf + 0.005, f'{delta:+.4f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    # Plot 5: Confidence vs R Value
    ax = axes[1, 1]
    r_conf = []
    r_conf_std = []
    for r in unique_r:
        configs = [res for res in all_results if abs(res['r_value'] - r) < 0.01]
        conf_vals = [c['avg_confidence'] for c in configs]
        r_conf.append(np.mean(conf_vals))
        r_conf_std.append(np.std(conf_vals))
    
    ax.errorbar(unique_r, r_conf, yerr=r_conf_std,
                marker='s', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkgreen', ecolor='lightgreen', label='Confidence')
    ax.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Confidence')
    ax.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax.set_ylabel('Confidence', fontweight='bold', fontsize=13)
    ax.set_title('Confidence vs R Value\n(averaged over saliency and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add confidence delta labels
    for i, (r, conf) in enumerate(zip(unique_r, r_conf)):
        delta = conf - avg_baseline_conf
        color = 'green' if delta >= -0.01 else 'red'
        ax.text(r, conf + 0.005, f'{delta:+.4f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    # Plot 6: Confidence vs Threshold
    ax = axes[1, 2]
    thresh_conf = []
    thresh_conf_std = []
    for t in unique_thresh:
        configs = [res for res in all_results if abs(res['threshold'] - t) < 0.01]
        conf_vals = [c['avg_confidence'] for c in configs]
        thresh_conf.append(np.mean(conf_vals))
        thresh_conf_std.append(np.std(conf_vals))
    
    ax.errorbar(unique_thresh, thresh_conf, yerr=thresh_conf_std,
                marker='^', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkorange', ecolor='orange', label='Confidence')
    ax.axhline(y=avg_baseline_conf, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Confidence')
    ax.set_xlabel('Matching Threshold', fontweight='bold', fontsize=13)
    ax.set_ylabel('Confidence', fontweight='bold', fontsize=13)
    ax.set_title('Confidence vs Threshold\n(averaged over saliency and r)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add confidence delta labels
    for i, (t, conf) in enumerate(zip(unique_thresh, thresh_conf)):
        delta = conf - avg_baseline_conf
        color = 'green' if delta >= -0.01 else 'red'
        ax.text(t, conf + 0.005, f'{delta:+.4f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    plt.suptitle(f'Comprehensive Hyperparameter Analysis\nBaseline: {avg_baseline_flops/1e6:.1f}M FLOPs, {avg_baseline_conf:.4f} Confidence\n({len(all_results)} configurations tested)',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_file4 = 'hyperparameter_analysis.png'
    plt.savefig(output_file4, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive hyperparameter analysis to: {output_file4}")
    
    output_pdf4 = 'hyperparameter_analysis.pdf'
    plt.savefig(output_pdf4, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf4}")
    
    # ========================================================================
    # Accuracy Analysis Plots (2x3 Grid: Top-1 and Top-5)
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING ACCURACY ANALYSIS PLOTS (Top-1 and Top-5)")
    print("="*80)
    
    fig5, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # ========= ROW 1: TOP-1 ACCURACY vs Parameters =========
    
    # Plot 1: Top-1 Accuracy vs Saliency Layers
    ax = axes[0, 0]
    saliency_top1 = []
    saliency_top1_std = []
    for sal in unique_saliency:
        configs = [r for r in all_results if r['num_saliency_layers'] == sal]
        top1_vals = [c['top1_accuracy']*100 for c in configs]
        saliency_top1.append(np.mean(top1_vals))
        saliency_top1_std.append(np.std(top1_vals))
    
    ax.errorbar(unique_saliency, saliency_top1, yerr=saliency_top1_std,
                marker='o', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkblue', ecolor='steelblue', label='Top-1 Accuracy (Cached)')
    ax.axhline(y=baseline_top1_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-1')
    ax.set_xlabel('Number of Saliency Layers', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-1 Accuracy vs Saliency Layers\n(averaged over r and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # Plot 2: Top-1 Accuracy vs R Value
    ax = axes[0, 1]
    r_top1 = []
    r_top1_std = []
    for r in unique_r:
        configs = [res for res in all_results if abs(res['r_value'] - r) < 0.01]
        top1_vals = [c['top1_accuracy']*100 for c in configs]
        r_top1.append(np.mean(top1_vals))
        r_top1_std.append(np.std(top1_vals))
    
    ax.errorbar(unique_r, r_top1, yerr=r_top1_std,
                marker='s', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkgreen', ecolor='lightgreen', label='Top-1 Accuracy (Cached)')
    ax.axhline(y=baseline_top1_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-1')
    ax.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-1 Accuracy vs R Value\n(averaged over saliency and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # Plot 3: Top-1 Accuracy vs Threshold
    ax = axes[0, 2]
    thresh_top1 = []
    thresh_top1_std = []
    for t in unique_thresh:
        configs = [res for res in all_results if abs(res['threshold'] - t) < 0.01]
        top1_vals = [c['top1_accuracy']*100 for c in configs]
        thresh_top1.append(np.mean(top1_vals))
        thresh_top1_std.append(np.std(top1_vals))
    
    ax.errorbar(unique_thresh, thresh_top1, yerr=thresh_top1_std,
                marker='^', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkorange', ecolor='orange', label='Top-1 Accuracy (Cached)')
    ax.axhline(y=baseline_top1_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-1')
    ax.set_xlabel('Matching Threshold', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-1 Accuracy vs Threshold\n(averaged over saliency and r)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # ========= ROW 2: TOP-5 ACCURACY vs Parameters =========
    
    # Plot 4: Top-5 Accuracy vs Saliency Layers
    ax = axes[1, 0]
    saliency_top5 = []
    saliency_top5_std = []
    for sal in unique_saliency:
        configs = [r for r in all_results if r['num_saliency_layers'] == sal]
        top5_vals = [c['top5_accuracy']*100 for c in configs]
        saliency_top5.append(np.mean(top5_vals))
        saliency_top5_std.append(np.std(top5_vals))
    
    ax.errorbar(unique_saliency, saliency_top5, yerr=saliency_top5_std,
                marker='o', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkblue', ecolor='steelblue', label='Top-5 Accuracy (Cached)')
    ax.axhline(y=baseline_top5_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-5')
    ax.set_xlabel('Number of Saliency Layers', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-5 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-5 Accuracy vs Saliency Layers\n(averaged over r and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # Plot 5: Top-5 Accuracy vs R Value
    ax = axes[1, 1]
    r_top5 = []
    r_top5_std = []
    for r in unique_r:
        configs = [res for res in all_results if abs(res['r_value'] - r) < 0.01]
        top5_vals = [c['top5_accuracy']*100 for c in configs]
        r_top5.append(np.mean(top5_vals))
        r_top5_std.append(np.std(top5_vals))
    
    ax.errorbar(unique_r, r_top5, yerr=r_top5_std,
                marker='s', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkgreen', ecolor='lightgreen', label='Top-5 Accuracy (Cached)')
    ax.axhline(y=baseline_top5_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-5')
    ax.set_xlabel('R Value', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-5 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-5 Accuracy vs R Value\n(averaged over saliency and threshold)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # Plot 6: Top-5 Accuracy vs Threshold
    ax = axes[1, 2]
    thresh_top5 = []
    thresh_top5_std = []
    for t in unique_thresh:
        configs = [res for res in all_results if abs(res['threshold'] - t) < 0.01]
        top5_vals = [c['top5_accuracy']*100 for c in configs]
        thresh_top5.append(np.mean(top5_vals))
        thresh_top5_std.append(np.std(top5_vals))
    
    ax.errorbar(unique_thresh, thresh_top5, yerr=thresh_top5_std,
                marker='^', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='darkorange', ecolor='orange', label='Top-5 Accuracy (Cached)')
    ax.axhline(y=baseline_top5_acc*100, color='red', linestyle='--', linewidth=3,
               alpha=0.7, label='Baseline Top-5')
    ax.set_xlabel('Matching Threshold', fontweight='bold', fontsize=13)
    ax.set_ylabel('Top-5 Accuracy (%)', fontweight='bold', fontsize=13)
    ax.set_title('Top-5 Accuracy vs Threshold\n(averaged over saliency and r)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.suptitle(f'Top-1 and Top-5 Accuracy Analysis\nBaseline: Top-1={baseline_top1_acc*100:.1f}%, Top-5={baseline_top5_acc*100:.1f}% | Ground Truth: {ground_truth_name}',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_file5 = 'accuracy_analysis.png'
    plt.savefig(output_file5, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy analysis to: {output_file5}")
    
    output_pdf5 = 'accuracy_analysis.pdf'
    plt.savefig(output_pdf5, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf5}")
    
    print("\n" + "="*80)
    print("K/V Caching Performance Analysis Complete!")
    print("="*80)
    print(f"\nGenerated plots:")
    print(f"  1. {output_file} - Main comparison (confidence & FLOPs)")
    print(f"  2. {output_file2} - R-value & threshold analysis (4 plots)")
    print(f"  3. {output_file3} - Best configurations summary")
    print(f"  4. {output_file4} - Comprehensive hyperparameter analysis (2x3 grid)")
    print(f"  5. {output_file5} - Accuracy analysis (Top-1 and Top-5, 2x3 grid)")
    
    print("\nKey Insights from Hyperparameter Analysis:")
    print(f"  • Saliency layers tested: {unique_saliency}")
    print(f"  • R values tested: {unique_r}")
    print(f"  • Thresholds tested: {unique_thresh}")
    print(f"  • Total configurations: {len(all_results)}")
    print(f"  • Best net FLOP reduction: {max([((baseline_ref - (r['avg_flops'] + r['avg_overhead']))/baseline_ref)*100 for r in all_results]):.2f}%")
    print(f"  • Average net FLOP reduction: {np.mean([((baseline_ref - (r['avg_flops'] + r['avg_overhead']))/baseline_ref)*100 for r in all_results]):.2f}%")
    
    print("\nAccuracy Summary:")
    print(f"  • Baseline Top-1 Accuracy: {baseline_top1_acc*100:.2f}%")
    print(f"  • Baseline Top-5 Accuracy: {baseline_top5_acc*100:.2f}%")
    print(f"  • Best Cached Top-1: {max([r['top1_accuracy'] for r in all_results])*100:.2f}%")
    print(f"  • Best Cached Top-5: {max([r['top5_accuracy'] for r in all_results])*100:.2f}%")
    print(f"  • Average Cached Top-1: {np.mean([r['top1_accuracy'] for r in all_results])*100:.2f}%")
    print(f"  • Average Cached Top-5: {np.mean([r['top5_accuracy'] for r in all_results])*100:.2f}%")


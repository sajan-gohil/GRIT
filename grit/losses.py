"""
Loss functions for training transformer layers with graph structure.

These losses capture attention improvements and structure reconstruction
during transformer encoding.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, to_undirected


def attention_improvement_loss(
    node_embeddings,
    denoised_embeddings,
    edge_index,
    batch,
    tau=0.2,
    weight=1.0
):
    """
    Attention improvement loss that encourages the transformer to maintain
    and improve structural associations.
    
    Args:
        node_embeddings: Initial node embeddings [N, D]
        denoised_embeddings: Embeddings after QKV transformation [N, D]
        edge_index: Edge indices [2, E]
        batch: Batch assignment for each node [N]
        tau: Temperature for sigmoid (default=0.2)
        weight: Scalar loss weight (default=1.0)
    
    Returns:
        Weighted scalar loss
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=node_embeddings.device)
    
    # Normalize embeddings for numerical stability
    node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
    denoised_embeddings = F.normalize(denoised_embeddings, p=2, dim=-1)
    
    # Compute similarity matrices for original and denoised embeddings
    # For positive edges (existing edges)
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    
    # Original similarities
    orig_sim = (node_embeddings[src_idx] * node_embeddings[dst_idx]).sum(dim=-1)
    
    # Denoised similarities
    denoised_sim = (denoised_embeddings[src_idx] * denoised_embeddings[dst_idx]).sum(dim=-1)
    
    # Sample negative edges (approximately 2x positive edges)
    num_neg_samples = edge_index.size(1) * 2
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=node_embeddings.size(0),
        num_neg_samples=num_neg_samples
    )
    
    neg_src_idx = neg_edge_index[0]
    neg_dst_idx = neg_edge_index[1]
    
    # Negative similarities
    orig_neg_sim = (node_embeddings[neg_src_idx] * node_embeddings[neg_dst_idx]).sum(dim=-1)
    denoised_neg_sim = (denoised_embeddings[neg_src_idx] * denoised_embeddings[neg_dst_idx]).sum(dim=-1)
    
    # Improvement: denoised should increase positive similarities and decrease negative ones
    # Positive edges: encourage higher similarity
    pos_improvement = torch.sigmoid((denoised_sim - orig_sim) / tau)
    
    # Negative edges: encourage lower similarity
    neg_improvement = torch.sigmoid((orig_neg_sim - denoised_neg_sim) / tau)
    
    # Compute per-graph loss
    num_graphs = batch.max().item() + 1
    loss = 0.0
    
    for graph_id in range(num_graphs):
        # Find nodes in this graph
        graph_mask = (batch == graph_id)
        graph_node_indices = torch.where(graph_mask)[0]
        
        # Find edges in this graph (both endpoints must be in the graph)
        pos_edge_mask = graph_mask[src_idx] & graph_mask[dst_idx]
        neg_edge_mask = graph_mask[neg_src_idx] & graph_mask[neg_dst_idx]
        
        if pos_edge_mask.sum() > 0:
            # Average improvement for positive edges
            graph_pos_loss = -torch.log(pos_improvement[pos_edge_mask].mean() + 1e-8)
            loss = loss + graph_pos_loss
        
        if neg_edge_mask.sum() > 0:
            # Average improvement for negative edges
            graph_neg_loss = -torch.log(neg_improvement[neg_edge_mask].mean() + 1e-8)
            loss = loss + graph_neg_loss
    
    # Normalize by number of graphs
    if num_graphs > 0:
        loss = loss / num_graphs
    
    return weight * loss


def structure_reconstruction_loss(
    node_embeddings,
    edge_index,
    batch,
    weight=1.0
):
    """
    Structure reconstruction loss that reconstructs graph structure from
    embeddings using predicted adjacency.
    
    Args:
        node_embeddings: Node embeddings after first transformer layer [N, D]
        edge_index: Original edge indices [2, E]
        batch: Batch assignment for each node [N]
        weight: Scalar loss weight (default=1.0)
    
    Returns:
        Weighted scalar loss
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=node_embeddings.device)
    
    # Normalize embeddings for numerical stability
    node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
    
    # Positive edges
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    
    # Compute similarity for positive edges (should be high)
    pos_scores = (node_embeddings[src_idx] * node_embeddings[dst_idx]).sum(dim=-1)
    
    # Sample negative edges (~2x positive samples)
    num_neg_samples = edge_index.size(1) * 2
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=node_embeddings.size(0),
        num_neg_samples=num_neg_samples
    )
    
    neg_src_idx = neg_edge_index[0]
    neg_dst_idx = neg_edge_index[1]
    
    # Compute similarity for negative edges (should be low)
    neg_scores = (node_embeddings[neg_src_idx] * node_embeddings[neg_dst_idx]).sum(dim=-1)
    
    # Compute per-graph loss with adaptive thresholds
    num_graphs = batch.max().item() + 1
    loss = 0.0
    
    for graph_id in range(num_graphs):
        # Find nodes in this graph
        graph_mask = (batch == graph_id)
        
        # Find edges in this graph (both endpoints must be in the graph)
        pos_edge_mask = graph_mask[src_idx] & graph_mask[dst_idx]
        neg_edge_mask = graph_mask[neg_src_idx] & graph_mask[neg_dst_idx]
        
        if pos_edge_mask.sum() > 0 and neg_edge_mask.sum() > 0:
            # Get scores for this graph
            graph_pos_scores = pos_scores[pos_edge_mask]
            graph_neg_scores = neg_scores[neg_edge_mask]
            
            # Compute per-graph threshold (median of positive and negative scores)
            threshold = (graph_pos_scores.median() + graph_neg_scores.median()) / 2.0
            
            # Binary cross-entropy style loss
            # Positive edges should have score > threshold
            pos_loss = F.binary_cross_entropy_with_logits(
                graph_pos_scores,
                torch.ones_like(graph_pos_scores),
                reduction='mean'
            )
            
            # Negative edges should have score < threshold
            neg_loss = F.binary_cross_entropy_with_logits(
                graph_neg_scores,
                torch.zeros_like(graph_neg_scores),
                reduction='mean'
            )
            
            graph_loss = (pos_loss + neg_loss) / 2.0
            loss = loss + graph_loss
    
    # Normalize by number of graphs
    if num_graphs > 0:
        loss = loss / num_graphs
    
    return weight * loss

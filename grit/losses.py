"""
Loss functions for training transformer layers with graph structure.

These losses capture attention improvements and structure reconstruction
during transformer encoding.
"""
import torch
import torch.nn.functional as F


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
    
    # Normalize embeddings using L2 normalization
    node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
    denoised_embeddings = F.normalize(denoised_embeddings, p=2, dim=-1)
    
    # Compute similarity scores for connected nodes before and after transformation
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    
    # Initial similarity scores
    initial_scores = (node_embeddings[src_idx] * node_embeddings[dst_idx]).sum(dim=-1)
    
    # Final similarity scores (after transformation)
    final_scores = (denoised_embeddings[src_idx] * denoised_embeddings[dst_idx]).sum(dim=-1)
    
    # Compute per-graph thresholds and losses
    num_graphs = batch.max().item() + 1
    loss = torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
    
    for graph_id in range(num_graphs):
        # Find nodes in this graph
        graph_mask = (batch == graph_id)
        
        # Find edges in this graph (both endpoints must be in the graph)
        edge_mask = graph_mask[src_idx] & graph_mask[dst_idx]
        
        if edge_mask.sum() > 0:
            # Get scores for this graph
            graph_initial_scores = initial_scores[edge_mask]
            graph_final_scores = final_scores[edge_mask]
            
            # Compute per-graph threshold as mean initial score
            threshold = graph_initial_scores.mean()
            
            # Use sigmoid to compute soft recall based on final score vs threshold
            # Higher final scores relative to threshold should give higher sigmoid values
            soft_recall = torch.sigmoid((graph_final_scores - threshold) / tau)
            
            # Loss: we want to maximize soft_recall, so minimize negative log
            graph_loss = -torch.log(soft_recall.mean() + 1e-8)
            
            # Accumulate loss normalized by edge count (implicitly via mean)
            loss = loss + graph_loss
    
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
    
    # Compute similarity scores for positive (connected) edges
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    pos_scores = (node_embeddings[src_idx] * node_embeddings[dst_idx]).sum(dim=-1)
    
    # Sample negative edges (~2x positive samples) from random node pairs in the same graph
    num_nodes = node_embeddings.size(0)
    num_pos_edges = edge_index.size(1)
    num_neg_samples = num_pos_edges * 2
    
    # Create a set of existing edges for filtering
    edge_set = set()
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        edge_set.add((src, dst))
        edge_set.add((dst, src))  # Also add reverse edge
    
    # Sample negative edges per graph
    neg_src_list = []
    neg_dst_list = []
    num_graphs = batch.max().item() + 1
    
    for graph_id in range(num_graphs):
        # Find nodes in this graph
        graph_mask = (batch == graph_id)
        graph_nodes = torch.where(graph_mask)[0]
        num_graph_nodes = graph_nodes.size(0)
        
        if num_graph_nodes < 2:
            continue
        
        # Count edges in this graph
        edge_mask = graph_mask[src_idx] & graph_mask[dst_idx]
        num_graph_edges = edge_mask.sum().item()
        
        # Sample ~2x the number of edges in this graph
        num_graph_neg_samples = num_graph_edges * 2
        
        sampled = 0
        max_attempts = num_graph_neg_samples * 10  # Prevent infinite loop
        attempts = 0
        
        while sampled < num_graph_neg_samples and attempts < max_attempts:
            # Sample random pairs from the graph
            batch_size = min(num_graph_neg_samples - sampled, 100)
            src_samples = graph_nodes[torch.randint(0, num_graph_nodes, (batch_size,))]
            dst_samples = graph_nodes[torch.randint(0, num_graph_nodes, (batch_size,))]
            
            for src, dst in zip(src_samples, dst_samples):
                src_item = src.item()
                dst_item = dst.item()
                
                # Filter out self-loops and actual edges
                if src_item != dst_item and (src_item, dst_item) not in edge_set:
                    neg_src_list.append(src_item)
                    neg_dst_list.append(dst_item)
                    sampled += 1
                    if sampled >= num_graph_neg_samples:
                        break
            
            attempts += batch_size
    
    # Handle case where we couldn't sample enough negatives
    if len(neg_src_list) == 0:
        return torch.tensor(0.0, device=node_embeddings.device)
    
    neg_src_idx = torch.tensor(neg_src_list, dtype=torch.long, device=node_embeddings.device)
    neg_dst_idx = torch.tensor(neg_dst_list, dtype=torch.long, device=node_embeddings.device)
    
    # Compute similarity scores for negative edges
    neg_scores = (node_embeddings[neg_src_idx] * node_embeddings[neg_dst_idx]).sum(dim=-1)
    
    # Compute BCE loss comparing sigmoid predictions to target values
    # For positive edges: target = 1
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores,
        torch.ones_like(pos_scores),
        reduction='mean'
    )
    
    # For negative edges: target = 0
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores,
        torch.zeros_like(neg_scores),
        reduction='mean'
    )
    
    # Return averaged loss across positive and negative samples
    loss = (pos_loss + neg_loss) / 2.0
    
    return weight * loss

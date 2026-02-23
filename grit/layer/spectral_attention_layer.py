import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.utils import to_dense_batch
from torch_geometric.graphgym.register import register_layer


# ==========================================
# 1. Core Component: Chebyshev Filter
# ==========================================
class ChebyshevFilter(nn.Module):
    """
    Applies a learnable spectral filter using Chebyshev polynomials.
    Output = Sum_k (theta_k * T_k(L_hat) * X)
    where L_hat = L_sym - I shifts eigenvalues from [0, 2] to [-1, 1].
    """

    def __init__(self, in_channels, K=3):
        super().__init__()
        self.K = K
        self.coeffs = nn.Parameter(torch.empty(K))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1)

    def forward(self, x, edge_index):
        """
        x: [N, dim]
        edge_index: [2, E]
        """
        num_nodes = x.size(0)

        # Add self-loops for Laplacian stability (remove existing ones first)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Compute normalised symmetric Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        edge_index_L, edge_weight_L = get_laplacian(
            edge_index_sl, normalization='sym', num_nodes=num_nodes
        )

        def sparse_mm(idx, wt, mat):
            return torch.sparse.mm(
                torch.sparse_coo_tensor(idx, wt, (num_nodes, num_nodes)),
                mat
            )

        # Chebyshev recurrence on L_hat = L_sym - I  (eigenvalues in [-1, 1])
        # T_0(L_hat) x = x
        # T_1(L_hat) x = L_hat x = L_sym x - x
        # T_k(L_hat) x = 2 L_hat T_{k-1} - T_{k-2}
        #              = 2 (L_sym T_{k-1} - T_{k-1}) - T_{k-2}

        Tx_0 = x
        Lx = sparse_mm(edge_index_L, edge_weight_L, x)
        Tx_1 = Lx - x  # = L_hat x

        out = self.coeffs[0] * Tx_0
        if self.K > 1:
            out = out + self.coeffs[1] * Tx_1

        Tx_prev2, Tx_prev = Tx_0, Tx_1
        for k in range(2, self.K):
            L_Tx_prev = sparse_mm(edge_index_L, edge_weight_L, Tx_prev)
            Tx_k = 2.0 * (L_Tx_prev - Tx_prev) - Tx_prev2
            out = out + self.coeffs[k] * Tx_k
            Tx_prev2, Tx_prev = Tx_prev, Tx_k

        return out


# ==========================================
# 2. Spectrally-Decoupled Attention (SDA) Layer
# ==========================================
class SDA_Layer(nn.Module):
    """
    Spectrally-Decoupled Attention layer.
    Queries and keys are filtered through per-head Chebyshev spectral filters
    before computing scaled dot-product attention.
    """

    def __init__(self, embed_dim, num_heads, K=3, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.filters_q = nn.ModuleList(
            [ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)]
        )
        self.filters_k = nn.ModuleList(
            [ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)]
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x, edge_index, batch_index):
        """
        x:           [N, embed_dim]
        edge_index:  [2, E]
        batch_index: [N]  batch assignment for each node
        Returns updated x: [N, embed_dim]
        """
        resid = x
        x = self.norm1(x)

        # Linear projections -> [N, num_heads, head_dim]
        q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.head_dim)

        # Spectral filtering per head
        q_spec_list, k_spec_list = [], []
        for h in range(self.num_heads):
            q_spec_list.append(self.filters_q[h](q[:, h, :], edge_index))
            k_spec_list.append(self.filters_k[h](k[:, h, :], edge_index))

        q_spec = torch.stack(q_spec_list, dim=1)  # [N, H, D]
        k_spec = torch.stack(k_spec_list, dim=1)  # [N, H, D]

        # Pack into dense padded batch tensors
        # to_dense_batch returns [B, max_N, feat_dim] and mask [B, max_N]
        q_dense, mask = to_dense_batch(
            q_spec.reshape(x.size(0), -1), batch_index
        )
        k_dense, _ = to_dense_batch(
            k_spec.reshape(x.size(0), -1), batch_index
        )
        v_dense, _ = to_dense_batch(
            v.reshape(x.size(0), -1), batch_index
        )

        B, max_N, _ = q_dense.size()
        # Reshape to [B, H, N, D]
        q_dense = q_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        k_dense = k_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        v_dense = v_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with padding mask
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_dense, k_dense.transpose(-2, -1)) * scale  # [B, H, N, N]

        # mask: True = real node; broadcast over heads and query positions
        mask_broadcast = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        scores = scores.masked_fill(~mask_broadcast, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out_dense = torch.matmul(attn, v_dense)  # [B, H, N, D]
        out_dense = out_dense.transpose(1, 2).reshape(B, max_N, self.embed_dim)

        # Unpack: select real (non-padded) nodes
        out = out_dense[mask]  # [N, embed_dim]
        out = self.W_o(out)

        # First residual
        x = resid + self.dropout(out)

        # FFN with second residual
        x = x + self.ffn(self.norm2(x))

        return x


# ==========================================
# 3. GRIT-compatible wrapper layer
# ==========================================
@register_layer("SpectralAttention")
class SpectralAttentionTransformerLayer(nn.Module):
    """
    Wraps SDA_Layer so it conforms to the GRIT framework's layer interface.
    The layer receives and returns a PyG Data batch object.
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.1,
                 layer_norm=False,
                 batch_norm=True,
                 residual=True,
                 act='relu',
                 cfg=None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim

        # Read Chebyshev order from config (default 3)
        K = 3
        if cfg is not None and hasattr(cfg, 'attn') and cfg.attn is not None:
            K = cfg.attn.get('K', 3)

        self.sda_layer = SDA_Layer(
            embed_dim=in_dim,
            num_heads=num_heads,
            K=K,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        batch.x = self.sda_layer(batch.x, batch.edge_index, batch.batch)
        if self.batch_norm:
            batch.x = self.bn(batch.x)
        return batch

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels})'
        )

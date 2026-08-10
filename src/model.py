import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from .utils import EDGE_DIM

class AdaptiveWeightedLaplacianLearning(nn.Module):
    def __init__(self, node_dim, hidden_dim, prompt_dim, edge_dim=EDGE_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.silu = nn.SiLU()

        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim // 2)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prompt_projection = nn.Linear(prompt_dim, hidden_dim)
        self.edge_attention = nn.Linear(hidden_dim // 2, 1)

        self.alpha = nn.Parameter(torch.tensor(0.001))
        self.beta = nn.Parameter(torch.tensor(0.0001))

    def compute_adaptive_weights(self, x, edge_index, edge_attr, h_p, h_i_proj):
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return torch.tensor([], device=x.device)

        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            if edge_attr.size(1) < self.edge_dim:
                padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(1), device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
            else:
                edge_attr = edge_attr[:, :self.edge_dim]

        src, dst = edge_index[0], edge_index[1]
        h_i = h_i_proj[src]
        h_j = h_i_proj[dst]

        h_i = self.layer_norm1(h_i)
        h_j = self.layer_norm2(h_j)

        h_j_W = self.W(h_j)
        structural = torch.sum(h_i * h_j_W, dim=1)
        structural = self.silu(structural)

        if h_p.dim() == 2 and h_p.size(0) == x.size(0):
            h_p_src = self.prompt_projection(h_p[src])
        else:
            h_p_avg = h_p.mean(dim=0, keepdim=True)
            h_p_proj = self.prompt_projection(h_p_avg)
            h_p_src = h_p_proj.expand(num_edges, -1)

        h_p_src = self.layer_norm3(h_p_src)
        h_sum = h_i + h_j
        h_sum_U = self.U(h_sum)
        semantic = torch.sum(h_p_src * h_sum_U, dim=1)
        semantic = self.silu(semantic)

        if edge_attr is not None and edge_attr.size(0) > 0:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(0)
            edge_features = self.edge_projection(edge_attr)
            edge_score = torch.sigmoid(self.edge_attention(edge_features).squeeze(-1))
            semantic = semantic * edge_score

        weights = torch.sigmoid(structural + semantic)
        return weights

    def forward(self, x, edge_index, edge_attr, h_p, original_adj=None, h_i_proj=None):
        device = x.device
        num_nodes = x.size(0)

        if h_i_proj is None:
            h_i_proj = self.node_projection(x)

        if edge_index.size(1) > 0:
            edge_weights = self.compute_adaptive_weights(x, edge_index, edge_attr, h_p, h_i_proj)
            adj_weighted = torch.zeros(num_nodes, num_nodes, device=device)
            adj_weighted[edge_index[0], edge_index[1]] = edge_weights
            adj_weighted = (adj_weighted + adj_weighted.T) / 2
        else:
            adj_weighted = torch.zeros(num_nodes, num_nodes, device=device)
            edge_weights = torch.tensor([], device=device)

        deg = adj_weighted.sum(dim=1) + 1e-8
        D_weighted = torch.diag(deg)
        L_weighted = D_weighted - adj_weighted

        if original_adj is not None:
            deg_orig = original_adj.sum(dim=1) + 1e-8
            D_orig = torch.diag(deg_orig)
            L_orig = D_orig - original_adj

            frob_norm = torch.mean((L_weighted - L_orig) ** 2)
            smooth_term = torch.mean(torch.diag(x.T @ L_weighted @ x))
            lap_loss = self.alpha * frob_norm + self.beta * smooth_term

            if h_p.dim() == 2:
                h_p_avg = h_p.mean(dim=0)
            else:
                h_p_avg = h_p

            h_p_proj = self.prompt_projection(h_p_avg.unsqueeze(0)).squeeze(0)
            prompt_loss = torch.mean((h_i_proj - h_p_proj.unsqueeze(0).expand_as(h_i_proj)) ** 2)

            return L_weighted, adj_weighted, edge_weights, lap_loss, prompt_loss

        return L_weighted, adj_weighted, edge_weights

class EdgeAwareGraphTransformerEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim=EDGE_DIM, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim

        self.input_projection = nn.Linear(node_dim, hidden_dim)
        self.input_layer_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            EdgeAwareGraphTransformerLayer(hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, adj_weighted):
        if x.size(1) != self.hidden_dim:
            H = self.input_projection(x)
        else:
            H = x

        H = self.input_layer_norm(H)

        deg = adj_weighted.sum(dim=1).unsqueeze(1)
        pos_encoding = deg * 0.1
        H = H + pos_encoding.expand(-1, self.hidden_dim)

        for layer in self.layers:
            H = layer(H, edge_index, edge_attr, adj_weighted)

        H = self.output_projection(H)
        H = self.layer_norm(H)
        return H

class EdgeAwareGraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.edge_projection = nn.Linear(edge_dim, num_heads)
        self.edge_layer_norm = nn.LayerNorm(hidden_dim)

        self.silu = nn.SiLU()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, edge_index, edge_attr, adj_weighted):
        attn_mask = (adj_weighted == 0).float() * -1e9
        attn_mask = torch.sigmoid(attn_mask) * -1e9

        attn_output, _ = self.multihead_attn(H, H, H, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        H = self.norm1(H + attn_output)

        if edge_index is not None and edge_index.size(1) > 0:
            if edge_attr is not None and edge_attr.size(0) > 0:
                if edge_attr.size(0) != edge_index.size(1):
                    edge_attr = torch.zeros(edge_index.size(1), self.edge_projection.in_features, device=H.device)
                elif edge_attr.size(1) != self.edge_projection.in_features:
                    if edge_attr.size(1) < self.edge_projection.in_features:
                        padding = torch.zeros(edge_attr.size(0), self.edge_projection.in_features - edge_attr.size(1), device=edge_attr.device)
                        edge_attr = torch.cat([edge_attr, padding], dim=1)
                    else:
                        edge_attr = edge_attr[:, :self.edge_projection.in_features]

                src, dst = edge_index[0], edge_index[1]
                H_src = H[src]

                edge_weights = torch.sigmoid(self.edge_projection(edge_attr))
                message = H_src * edge_weights.mean(dim=1, keepdim=True)

                aggregated = torch.zeros_like(H)
                aggregated = aggregated.scatter_add(0, dst.unsqueeze(1).expand(-1, H.size(1)), message)

                aggregated = self.edge_layer_norm(aggregated)
                H = H + aggregated

        ffn_output = self.ffn(H)
        H = self.norm2(H + ffn_output)
        return H

class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch=None):
        if batch is None:
            weights = self.attention(x)
            weights = torch.softmax(weights, dim=0)
            return (x * weights).sum(dim=0, keepdim=True)
        else:
            weights = self.attention(x)
            max_batch = batch.max().item() + 1
            pooled = []
            for b in range(max_batch):
                mask = (batch == b)
                if mask.sum() > 0:
                    w = weights[mask]
                    w = torch.softmax(w, dim=0)
                    pooled.append((x[mask] * w).sum(dim=0, keepdim=True))

            if len(pooled) > 0:
                return torch.cat(pooled, dim=0)
            else:
                return torch.zeros(0, x.size(1), device=x.device)

class MultiScaleReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_pool = GlobalAttentionPooling(hidden_dim)
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.add_pool = global_add_pool

        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch=None):
        attn_pool = self.attention_pool(x, batch)
        mean_pool = self.mean_pool(x, batch)
        max_pool = self.max_pool(x, batch)
        add_pool = self.add_pool(x, batch)

        combined = torch.cat([attn_pool, mean_pool, max_pool, add_pool], dim=1)
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)

        return fused

class EnhancedPromptLapFormer(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_heads=8, num_layers=3,
                 num_prompts=5, prompt_dim=256, dropout=0.1, edge_dim=EDGE_DIM, descriptor_dim=11):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers
        self.descriptor_dim = descriptor_dim
        self.edge_dim = edge_dim

        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.1)
        self.prompt_attention = nn.MultiheadAttention(prompt_dim, num_heads, batch_first=True)

        self.awll = AdaptiveWeightedLaplacianLearning(node_dim, hidden_dim, prompt_dim, edge_dim)

        self.graph_transformer = EdgeAwareGraphTransformerEncoder(
            node_dim, hidden_dim, edge_dim, num_heads, num_layers, dropout
        )

        self.multiscale_readout = MultiScaleReadout(hidden_dim)

        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def get_trainable_prompts(self, h_p_precomputed=None, num_nodes=None, batch=None):
        if h_p_precomputed is not None:
            return h_p_precomputed

        prompts = self.prompts.unsqueeze(0)
        prompts, _ = self.prompt_attention(prompts, prompts, prompts)
        prompts = prompts.mean(dim=1, keepdim=True)

        if num_nodes is not None:
            if batch is not None:
                batch_size = batch.max().item() + 1
                prompts = prompts.expand(batch_size, -1, -1)
                prompts = prompts.squeeze(1)
                return prompts
            else:
                prompts = prompts.expand(num_nodes, -1, -1)
                prompts = prompts.squeeze(1)
                return prompts

        return prompts.squeeze(0).squeeze(0)

    def forward(self, data, return_all_losses=False, h_p_precomputed=None):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        descriptors = data.descriptors if hasattr(data, 'descriptors') else None

        if edge_index.dim() == 1:
            edge_index = edge_index.reshape(2, -1)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(0)
            if edge_attr.size(1) != self.edge_dim:
                if edge_attr.size(1) < self.edge_dim:
                    padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(1), device=edge_attr.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=1)
                else:
                    edge_attr = edge_attr[:, :self.edge_dim]
        else:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=x.device)

        num_nodes = x.size(0)
        original_adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            original_adj[edge_index[0], edge_index[1]] = 1.0

        h_p = self.get_trainable_prompts(h_p_precomputed, num_nodes, batch)

        if h_p.dim() == 2:
            if h_p.size(0) == 1:
                h_p = h_p.expand(num_nodes, -1)
            elif h_p.size(0) != num_nodes and batch is not None:
                h_p_expanded = torch.zeros(num_nodes, self.prompt_dim, device=x.device)
                unique_batches = torch.unique(batch)
                for i, b in enumerate(unique_batches):
                    mask = (batch == b)
                    if i < h_p.size(0):
                        h_p_expanded[mask] = h_p[i]
                    else:
                        h_p_expanded[mask] = h_p[0] if h_p.size(0) > 0 else torch.zeros(self.prompt_dim, device=x.device)
                h_p = h_p_expanded

        h_p = torch.nan_to_num(h_p, nan=0.0, posinf=1.0, neginf=-1.0)

        h_i_proj = self.node_projection(x)
        h_i_proj = self.layer_norm(h_i_proj)

        L_weighted, adj_weighted, edge_weights, lap_loss, prompt_loss = self.awll(
            x, edge_index, edge_attr, h_p, original_adj, h_i_proj
        )

        H_star = self.graph_transformer(h_i_proj, edge_index, edge_attr, adj_weighted)

        graph_rep = self.multiscale_readout(H_star, batch)

        if descriptors is not None:
            desc_rep = self.descriptor_encoder(descriptors)
            combined = torch.cat([graph_rep, desc_rep], dim=1)
            combined = self.fusion(combined)
        else:
            combined = graph_rep

        logits = self.prediction_head(combined)

        if return_all_losses:
            return {
                'logits': logits,
                'lap_loss': lap_loss,
                'prompt_loss': prompt_loss,
                'graph_rep': graph_rep,
                'edge_weights': edge_weights
            }

        return logits

"""
Graph Neural Network Module for SGT.

Builds storm interaction graph and applies GNN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


class StormGraphBuilder:
    """
    Constructs graph from storm node positions.
    Uses k-NN connectivity based on spatial proximity.

    Args:
        k_neighbors: Number of nearest neighbors (default: 8)
        max_distance: Maximum edge distance in pixels (default: None = no limit)
    """

    def __init__(self, k_neighbors=8, max_distance=None):
        self.k = k_neighbors
        self.max_distance = max_distance

    def build_graph(self, node_positions_list, node_features_list, batch_idx):
        """
        Build graph from node positions.

        Args:
            node_positions_list: List of (N_i, 2) position tensors
            node_features_list: List of (N_i, C) feature tensors
            batch_idx: (N_total,) batch assignment tensor

        Returns:
            data: PyG Data object with edge_index, edge_attr, node features
        """
        device = node_features_list[0].device

        # Concatenate all nodes
        all_positions = torch.cat(node_positions_list, dim=0)  # (N_total, 2)
        all_features = torch.cat(node_features_list, dim=0)  # (N_total, C)

        # Build edges using k-NN within each batch
        edge_index_list = []
        edge_attr_list = []

        batch_size = batch_idx.max().item() + 1
        offset = 0

        for b in range(batch_size):
            # Get nodes for this batch
            mask = batch_idx == b
            batch_positions = all_positions[mask]  # (N_b, 2)
            N_b = batch_positions.shape[0]

            if N_b <= 1:
                # Single node - add self-loop
                edge_index_list.append(torch.tensor([[offset], [offset]], device=device))
                edge_attr_list.append(torch.zeros(1, 3, device=device))
                offset += N_b
                continue

            # Compute pairwise distances
            dist_matrix = torch.cdist(batch_positions, batch_positions)  # (N_b, N_b)

            # k-NN edges
            k = min(self.k, N_b - 1)
            _, knn_indices = torch.topk(dist_matrix, k + 1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]  # Exclude self (distance=0)

            # Build edge list
            sources = torch.arange(N_b, device=device).unsqueeze(1).expand(-1, k).flatten()
            targets = knn_indices.flatten()

            # Add global offset for batching
            sources = sources + offset
            targets = targets + offset

            edges = torch.stack([sources, targets], dim=0)  # (2, N_b * k)

            # Edge attributes: [distance, dx, dy]
            edge_attr = []
            for i in range(N_b):
                for j in knn_indices[i]:
                    pos_i = batch_positions[i]
                    pos_j = batch_positions[j]

                    dx = pos_j[0] - pos_i[0]
                    dy = pos_j[1] - pos_i[1]
                    dist = torch.sqrt(dx**2 + dy**2)

                    edge_attr.append(torch.tensor([dist, dx, dy], device=device))

            edge_attr = torch.stack(edge_attr)  # (N_b * k, 3)

            edge_index_list.append(edges)
            edge_attr_list.append(edge_attr)

            offset += N_b

        # Concatenate all edges
        edge_index = torch.cat(edge_index_list, dim=1)  # (2, E_total)
        edge_attr = torch.cat(edge_attr_list, dim=0)  # (E_total, 3)

        # Create PyG Data object
        data = Data(
            x=all_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_idx,
            pos=all_positions
        )

        return data


class StormGNN(nn.Module):
    """
    Graph Neural Network for modeling storm interactions.
    Uses Graph Attention Networks (GAT).

    Args:
        hidden_dim: Feature dimension (default: 128)
        num_layers: Number of GNN layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=num_heads  # Use edge features in attention
                )
            )

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Apply GNN to graph.

        Args:
            data: PyG Data object with x, edge_index, edge_attr

        Returns:
            updated_features: (N, hidden_dim) updated node features
            attention_weights: List of attention weight tensors (for visualization)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode edge features for attention
        edge_feat = self.edge_encoder(edge_attr)  # (E, num_heads)

        attention_weights = []

        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            # GAT layer with edge features
            x_new, (edge_idx, attn) = conv(
                x, edge_index, edge_attr=edge_feat, return_attention_weights=True
            )

            # Residual + norm
            x = self.norms[i](x + self.dropout(x_new))

            # Store attention for visualization
            attention_weights.append((edge_idx, attn))

        return x, attention_weights


class GraphToGrid(nn.Module):
    """
    Project graph node features back to spatial grid.
    Uses Gaussian splatting (learnable sigma).

    Args:
        hidden_dim: Feature dimension
        grid_size: Output grid size (default: 96)
        sigma: Gaussian kernel width (default: 4.0, learnable)
    """

    def __init__(self, hidden_dim=128, grid_size=96, sigma=4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        # Learnable Gaussian width
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))

    def forward(self, node_features, node_positions, batch_idx, batch_size):
        """
        Splat node features onto grid.

        Args:
            node_features: (N, hidden_dim)
            node_positions: (N, 2) in pixel coordinates
            batch_idx: (N,) batch assignment
            batch_size: Number of samples in batch

        Returns:
            grid_features: (B, hidden_dim, H, W)
        """
        device = node_features.device
        H, W = self.grid_size, self.grid_size

        # Create output grid
        grid = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)

        # Grid coordinates
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([grid_y, grid_x], dim=-1)  # (H, W, 2)

        sigma = torch.exp(self.log_sigma)

        # Splat each node
        for b in range(batch_size):
            mask = batch_idx == b
            batch_feats = node_features[mask]  # (N_b, C)
            batch_pos = node_positions[mask]  # (N_b, 2)

            for i in range(batch_feats.shape[0]):
                pos = batch_pos[i]  # (2,)
                feat = batch_feats[i]  # (C,)

                # Compute Gaussian weights
                dist = torch.sqrt(
                    (grid_coords[..., 0] - pos[0])**2 +
                    (grid_coords[..., 1] - pos[1])**2
                )  # (H, W)

                weights = torch.exp(-dist**2 / (2 * sigma**2))  # (H, W)

                # Normalize weights
                weights = weights / (weights.sum() + 1e-8)

                # Splat features
                grid[b] += weights.unsqueeze(0) * feat.unsqueeze(-1).unsqueeze(-1)

        return grid

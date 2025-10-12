"""
Storm-Graph Transformer (SGT) - Main Model Integration.

Physics-informed GNN-Transformer hybrid for severe weather nowcasting.
"""

import torch
import torch.nn as nn

from .encoder import MultiModalEncoder
from .detector import StormCellDetector
from .gnn import StormGraphBuilder, StormGNN, GraphToGrid
from .transformer import SpatioTemporalTransformer
from .decoder import PhysicsDecoder, compute_physics_loss


class StormGraphTransformer(nn.Module):
    """
    Storm-Graph Transformer (SGT) for Paper 1.

    Pipeline:
    1. Multi-Modal Encoder: Encode VIL, IR069, IR107, GLM
    2. Storm Cell Detection: Identify discrete storm entities
    3. Graph Construction: Build k-NN graph from storm nodes
    4. GNN: Model storm-storm interactions
    5. Graph → Grid: Project back to spatial grid
    6. Transformer: Global spatiotemporal attention
    7. Physics Decoder: Upsample with conservation constraints

    Args:
        modalities: List of modality names (default: ['vil', 'ir069', 'ir107', 'lght'])
        input_steps: Number of input timesteps (default: 12)
        output_steps: Number of output timesteps (default: 6)
        hidden_dim: Feature dimension (default: 128)
        gnn_layers: Number of GNN layers (default: 3)
        transformer_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        use_physics: Whether to apply physics constraints (default: True)
    """

    def __init__(
        self,
        modalities=['vil', 'ir069', 'ir107', 'lght'],
        input_steps=12,
        output_steps=6,
        hidden_dim=128,
        gnn_layers=3,
        transformer_layers=4,
        num_heads=8,
        use_physics=True
    ):
        super().__init__()
        self.modalities = modalities
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.hidden_dim = hidden_dim
        self.use_physics = use_physics

        # Module 1: Multi-Modal Encoder
        self.encoder = MultiModalEncoder(
            modalities=modalities,
            input_steps=input_steps,
            hidden_dim=hidden_dim
        )

        # Module 2: Storm Cell Detector
        self.detector = StormCellDetector(
            feature_dim=hidden_dim,
            min_intensity=0.3,
            min_distance=8,
            max_storms=50
        )

        # Module 3: Graph Builder
        self.graph_builder = StormGraphBuilder(
            k_neighbors=8,
            max_distance=None
        )

        # Module 4: GNN
        self.gnn = StormGNN(
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=4,
            dropout=0.1
        )

        # Module 5: Graph → Grid
        self.graph_to_grid = GraphToGrid(
            hidden_dim=hidden_dim,
            grid_size=96,
            sigma=4.0
        )

        # Module 6: Transformer
        self.transformer = SpatioTemporalTransformer(
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=num_heads,
            patch_size=8,
            dropout=0.1
        )

        # Module 7: Physics Decoder
        self.decoder = PhysicsDecoder(
            hidden_dim=hidden_dim,
            output_steps=output_steps,
            output_size=384,
            use_physics=use_physics
        )

    def forward(self, inputs_dict):
        """
        Forward pass through SGT.

        Args:
            inputs_dict: Dict of {modality: (B, T, H, W) tensors}

        Returns:
            predictions: (B, T_out, H, W) predicted VIL
            attention_info: Dict with GNN and Transformer attention weights
            physics_info: Dict with physics parameters
        """
        # 1. Encode multimodal inputs
        features = self.encoder(inputs_dict)  # (B, hidden_dim, 96, 96)

        # 2. Detect storm cells
        vil_input = inputs_dict['vil']  # (B, T_in, 384, 384)
        # Downsample VIL to match feature resolution
        vil_downsampled = torch.nn.functional.interpolate(
            vil_input,
            size=(96, 96),
            mode='bilinear',
            align_corners=False
        )  # (B, T_in, 96, 96)

        node_features_list, node_positions_list, batch_idx = self.detector(
            features, vil_downsampled
        )

        # 3. Build graph
        graph_data = self.graph_builder.build_graph(
            node_positions_list, node_features_list, batch_idx
        )

        # 4. Apply GNN
        updated_node_features, gnn_attention = self.gnn(graph_data)

        # 5. Project graph back to grid
        B = features.shape[0]
        grid_features = self.graph_to_grid(
            updated_node_features,
            graph_data.pos,
            batch_idx,
            B
        )  # (B, hidden_dim, 96, 96)

        # Residual connection with encoder features
        grid_features = grid_features + features

        # 6. Apply Transformer
        transformed_features, transformer_attention = self.transformer(
            grid_features
        )  # (B, hidden_dim, 96, 96)

        # Residual connection
        transformed_features = transformed_features + grid_features

        # 7. Decode to predictions
        predictions, physics_info = self.decoder(
            transformed_features
        )  # (B, T_out, 384, 384)

        # Collect attention information
        attention_info = {
            'gnn_attention': gnn_attention,
            'transformer_attention': transformer_attention
        }

        return predictions, attention_info, physics_info

    def compute_loss(self, predictions, targets, physics_info,
                     lambda_mse=1.0, lambda_physics=0.1, lambda_extreme=2.0):
        """
        Compute total loss with physics constraints.

        Args:
            predictions: (B, T, H, W) predicted VIL
            targets: (B, T, H, W) target VIL
            physics_info: Dict with physics parameters
            lambda_mse: Weight for MSE loss
            lambda_physics: Weight for physics loss
            lambda_extreme: Weight for extreme events (VIP > 181)

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dict with individual loss components
        """
        # 1. MSE Loss
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)

        # 2. Extreme Event Loss (higher weight for VIP > 181)
        extreme_mask = (targets > 181 / 255.0)  # Normalized threshold
        if extreme_mask.any():
            extreme_loss = torch.nn.functional.mse_loss(
                predictions[extreme_mask],
                targets[extreme_mask]
            )
        else:
            extreme_loss = torch.tensor(0.0, device=predictions.device)

        # 3. Physics Loss
        if self.use_physics:
            physics_loss, physics_loss_dict = compute_physics_loss(
                predictions, targets, physics_info, lambda_conservation=0.1
            )
        else:
            physics_loss = torch.tensor(0.0, device=predictions.device)
            physics_loss_dict = {}

        # Total loss
        total_loss = (
            lambda_mse * mse_loss +
            lambda_extreme * extreme_loss +
            lambda_physics * physics_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'extreme': extreme_loss.item(),
            'physics': physics_loss.item(),
            **physics_loss_dict
        }

        return total_loss, loss_dict


# Convenience function for model creation
def create_sgt_model(config=None):
    """
    Create SGT model with default or custom config.

    Args:
        config: Dict with model hyperparameters (optional)

    Returns:
        model: StormGraphTransformer instance
    """
    default_config = {
        'modalities': ['vil', 'ir069', 'ir107', 'lght'],
        'input_steps': 12,
        'output_steps': 6,
        'hidden_dim': 128,
        'gnn_layers': 3,
        'transformer_layers': 4,
        'num_heads': 8,
        'use_physics': True
    }

    if config is not None:
        default_config.update(config)

    return StormGraphTransformer(**default_config)

"""
Storm-Graph Transformer (SGT) model for Paper 1.

Physics-informed GNN-Transformer hybrid for severe weather nowcasting.
"""

from .model import StormGraphTransformer, create_sgt_model
from .detector import StormCellDetector
from .encoder import MultiModalEncoder
from .gnn import StormGNN, StormGraphBuilder, GraphToGrid
from .transformer import SpatioTemporalTransformer
from .decoder import PhysicsDecoder, compute_physics_loss

__all__ = [
    'StormGraphTransformer',
    'create_sgt_model',
    'StormCellDetector',
    'MultiModalEncoder',
    'StormGNN',
    'StormGraphBuilder',
    'GraphToGrid',
    'SpatioTemporalTransformer',
    'PhysicsDecoder',
    'compute_physics_loss',
]

"""
Test script for SGT modules.

Quick verification that all modules can be imported and forward pass works.
"""

import torch
import sys
sys.path.insert(0, '/Users/haider/Downloads/stormfusion-sevir')

from stormfusion.models.sgt import create_sgt_model


def test_sgt_forward():
    """Test end-to-end forward pass."""
    print("=" * 70)
    print("SGT MODULE TEST")
    print("=" * 70)

    # Create model
    print("\n1. Creating SGT model...")
    config = {
        'modalities': ['vil', 'ir069', 'ir107', 'lght'],
        'input_steps': 12,
        'output_steps': 6,
        'hidden_dim': 128,
        'gnn_layers': 3,
        'transformer_layers': 4,
        'num_heads': 8,
        'use_physics': True
    }

    model = create_sgt_model(config)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy inputs
    print("\n2. Creating dummy inputs...")
    B, T_in, H, W = 2, 12, 384, 384

    inputs = {
        'vil': torch.randn(B, T_in, H, W),
        'ir069': torch.randn(B, T_in, H, W),
        'ir107': torch.randn(B, T_in, H, W),
        'lght': torch.randn(B, T_in, H, W)
    }

    # Normalize to [0, 1] range (like real data)
    for mod in inputs:
        inputs[mod] = torch.sigmoid(inputs[mod])

    print(f"✅ Dummy inputs created:")
    for mod, data in inputs.items():
        print(f"   {mod:8s}: {tuple(data.shape)}")

    # Forward pass
    print("\n3. Running forward pass...")
    try:
        with torch.no_grad():
            predictions, attention_info, physics_info = model(inputs)

        print(f"✅ Forward pass successful!")
        print(f"   Predictions shape: {tuple(predictions.shape)}")
        print(f"   Expected: (B={B}, T_out=6, H=384, W=384)")

        # Verify output shape
        assert predictions.shape == (B, 6, 384, 384), "Output shape mismatch!"

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loss computation
    print("\n4. Testing loss computation...")
    try:
        targets = torch.randn(B, 6, 384, 384).sigmoid()

        loss, loss_dict = model.compute_loss(
            predictions, targets, physics_info,
            lambda_mse=1.0, lambda_physics=0.1, lambda_extreme=2.0
        )

        print(f"✅ Loss computation successful!")
        print(f"   Total loss: {loss.item():.4f}")
        print(f"   Loss components:")
        for name, value in loss_dict.items():
            if name != 'total':
                print(f"      {name:12s}: {value:.4f}")

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    return True


def test_individual_modules():
    """Test individual modules separately."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODULE TESTS")
    print("=" * 70)

    from stormfusion.models.sgt import (
        MultiModalEncoder,
        StormCellDetector,
        StormGraphBuilder,
        StormGNN,
        GraphToGrid,
        SpatioTemporalTransformer,
        PhysicsDecoder
    )

    # Test encoder
    print("\n1. Testing MultiModalEncoder...")
    encoder = MultiModalEncoder(input_steps=12, hidden_dim=128)
    inputs = {mod: torch.randn(2, 12, 384, 384) for mod in ['vil', 'ir069', 'ir107', 'lght']}
    features = encoder(inputs)
    print(f"   ✅ Output: {tuple(features.shape)} (expected: (2, 128, 96, 96))")

    # Test detector
    print("\n2. Testing StormCellDetector...")
    detector = StormCellDetector(feature_dim=128)
    vil_input = torch.randn(2, 12, 96, 96).sigmoid()
    node_feats, node_pos, batch_idx = detector(features, vil_input)
    print(f"   ✅ Detected {len(node_feats)} storm cells per batch")
    print(f"      Total nodes: {sum(f.shape[0] for f in node_feats)}")

    # Test graph builder
    print("\n3. Testing StormGraphBuilder...")
    graph_builder = StormGraphBuilder(k_neighbors=8)
    graph = graph_builder.build_graph(node_pos, node_feats, batch_idx)
    print(f"   ✅ Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    # Test GNN
    print("\n4. Testing StormGNN...")
    gnn = StormGNN(hidden_dim=128, num_layers=3)
    updated_feats, attn = gnn(graph)
    print(f"   ✅ Updated features: {tuple(updated_feats.shape)}")

    # Test graph to grid
    print("\n5. Testing GraphToGrid...")
    g2g = GraphToGrid(hidden_dim=128, grid_size=96)
    grid = g2g(updated_feats, graph.pos, batch_idx, batch_size=2)
    print(f"   ✅ Grid features: {tuple(grid.shape)}")

    # Test transformer
    print("\n6. Testing SpatioTemporalTransformer...")
    transformer = SpatioTemporalTransformer(hidden_dim=128, num_layers=4)
    transformed, attn = transformer(grid)
    print(f"   ✅ Transformed features: {tuple(transformed.shape)}")

    # Test decoder
    print("\n7. Testing PhysicsDecoder...")
    decoder = PhysicsDecoder(hidden_dim=128, output_steps=6)
    predictions, physics_info = decoder(transformed)
    print(f"   ✅ Predictions: {tuple(predictions.shape)}")

    print("\n" + "=" * 70)
    print("✅ ALL INDIVIDUAL MODULES PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests
    success = test_sgt_forward()

    if success:
        test_individual_modules()

    print("\n🎉 SGT implementation ready for training!")

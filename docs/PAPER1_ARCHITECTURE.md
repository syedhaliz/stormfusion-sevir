# Paper 1: Storm-Graph Transformer (SGT) Architecture

**Title:** "Physics-Informed Graph Neural Networks with Transformers for Severe Weather Nowcasting"

**Status:** Design Phase
**Timeline:** Week 1-3 (Oct 10-31)
**Target:** ArXiv + NeurIPS workshop / ICLR tiny papers

---

## Core Innovation

**Problem:** Existing deep learning models for weather nowcasting:
1. Treat weather as continuous fields (CNNs) - miss discrete storm structure
2. Ignore physical constraints - prone to hallucination
3. Limited interpretability - black box predictions

**Our Solution:** Storm-Graph Transformer (SGT)
1. **Graph Neural Network:** Models storms as discrete interacting entities
2. **Transformer:** Captures long-range spatiotemporal dependencies
3. **Physics Constraints:** Enforces conservation laws and atmospheric dynamics

**Key Insight:** Severe weather is dominated by discrete convective cells that interact, merge, split, and advect. Treating these as graph nodes enables:
- Explicit modeling of storm-storm interactions
- Interpretable attention (which storms influence prediction?)
- Natural incorporation of physics (advection graphs, merging rules)

---

## Architecture Design

### 1. Multi-Modal Encoder

**Input:** 4 SEVIR modalities × 12 timesteps
- VIL (radar): (B, 12, 384, 384)
- IR069 (water vapor): (B, 12, 384, 384)
- IR107 (IR window): (B, 12, 384, 384)
- GLM (lightning): (B, 12, 384, 384)

**Approach:** Separate CNN encoders per modality, then fuse

```python
class MultiModalEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        # Per-modality encoders (ResNet-like)
        self.vil_encoder = ResNetEncoder(in_channels=12, out_dim=hidden_dim)
        self.ir069_encoder = ResNetEncoder(in_channels=12, out_dim=hidden_dim)
        self.ir107_encoder = ResNetEncoder(in_channels=12, out_dim=hidden_dim)
        self.glm_encoder = ResNetEncoder(in_channels=12, out_dim=hidden_dim)

        # Fusion layer
        self.fusion = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)

    def forward(self, vil, ir069, ir107, glm):
        # Encode each modality
        f_vil = self.vil_encoder(vil)      # (B, hidden_dim, H', W')
        f_ir069 = self.ir069_encoder(ir069)
        f_ir107 = self.ir107_encoder(ir107)
        f_glm = self.glm_encoder(glm)

        # Concatenate and fuse
        f_all = torch.cat([f_vil, f_ir069, f_ir107, f_glm], dim=1)
        f_fused = self.fusion(f_all)  # (B, hidden_dim, H', W')

        return f_fused
```

**Novelty:** Modality-specific encoders preserve unique characteristics (radar vs IR vs lightning)

---

### 2. Storm Cell Detection

**Goal:** Identify discrete storm cells from encoded features

**Approach:** Watershed segmentation + feature extraction

```python
class StormCellDetector(nn.Module):
    def __init__(self, threshold=0.3, min_size=100):
        self.threshold = threshold
        self.min_size = min_size

    def forward(self, features, vil_input):
        """
        Args:
            features: (B, C, H, W) encoded features
            vil_input: (B, 12, 384, 384) raw VIL for peak detection

        Returns:
            nodes: List of (B,) tensors, each (N_i, C) node features
            positions: List of (B,) tensors, each (N_i, 2) (x, y) positions
        """
        # Use latest VIL frame for peak detection
        vil_latest = vil_input[:, -1]  # (B, 384, 384)

        nodes_batch = []
        positions_batch = []

        for b in range(features.size(0)):
            # Detect storm cells (peaks above threshold)
            peaks = self.detect_peaks(vil_latest[b], self.threshold)

            # Extract features at peak locations
            node_feats = self.extract_node_features(
                features[b], peaks, radius=16
            )

            nodes_batch.append(node_feats)
            positions_batch.append(peaks)

        return nodes_batch, positions_batch
```

**Novelty:** Dynamic graph construction - number of nodes varies per sample

---

### 3. Graph Construction

**Goal:** Build adjacency matrix based on spatial proximity

**Approach:** k-NN or radius-based graph

```python
class StormGraphBuilder:
    def __init__(self, k_neighbors=8, max_distance=50):
        self.k = k_neighbors
        self.max_dist = max_distance

    def build_graph(self, positions):
        """
        Args:
            positions: (N, 2) storm cell positions

        Returns:
            edge_index: (2, E) edge list
            edge_attr: (E, D) edge features (distance, direction)
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (N, N)

        # k-NN graph
        _, indices = dist_matrix.topk(self.k + 1, largest=False, dim=1)
        indices = indices[:, 1:]  # Remove self-loops

        # Build edge list
        edge_index = []
        edge_attr = []

        for i in range(positions.size(0)):
            for j in indices[i]:
                if dist_matrix[i, j] < self.max_dist:
                    edge_index.append([i, j])

                    # Edge features: distance + direction
                    dist = dist_matrix[i, j]
                    direction = (positions[j] - positions[i]) / (dist + 1e-6)
                    edge_attr.append(torch.cat([dist.unsqueeze(0), direction]))

        edge_index = torch.tensor(edge_index).t()  # (2, E)
        edge_attr = torch.stack(edge_attr)  # (E, 3)

        return edge_index, edge_attr
```

**Novelty:** Edge features encode spatial relationships (advection priors)

---

### 4. GNN Module

**Goal:** Model storm-storm interactions (merging, splitting, influence)

**Approach:** Graph Attention Network (GAT) or Message Passing

```python
class StormGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, edge_dim=3)
            for _ in range(num_layers)
        ])

    def forward(self, node_features, edge_index, edge_attr):
        """
        Args:
            node_features: (N, hidden_dim)
            edge_index: (2, E)
            edge_attr: (E, 3) [distance, dx, dy]

        Returns:
            updated_features: (N, hidden_dim)
        """
        x = node_features

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        return x  # (N, hidden_dim)
```

**Novelty:**
- Attention weights show which storms influence each other
- Edge attributes encode physical advection direction

**Physics Insight:**
- Nearby storms can merge (high mutual attention)
- Upstream storms advect toward downstream (directional edges)

---

### 5. Grid Projection (Graph → Image)

**Goal:** Convert graph representation back to spatial grid

**Approach:** Learnable projection with Gaussian splatting

```python
class GraphToGrid(nn.Module):
    def __init__(self, grid_size=(96, 96), hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, positions):
        """
        Args:
            node_features: (N, hidden_dim)
            positions: (N, 2) in [0, 1] normalized coordinates

        Returns:
            grid_features: (hidden_dim, H, W)
        """
        H, W = self.grid_size
        grid = torch.zeros(node_features.size(1), H, W).to(node_features.device)

        # Project node features
        projected = self.projection(node_features)  # (N, hidden_dim)

        # Gaussian splat onto grid
        for i in range(node_features.size(0)):
            pos = positions[i] * torch.tensor([W, H])  # Scale to grid
            x, y = int(pos[0]), int(pos[1])

            # Place features with Gaussian kernel
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        weight = torch.exp(-0.5 * (dx**2 + dy**2) / (4**2))
                        grid[:, ny, nx] += weight * projected[i]

        return grid
```

---

### 6. Transformer Module

**Goal:** Capture global spatiotemporal dependencies

**Approach:** Vision Transformer on grid features

```python
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(hidden_dim, patch_size=8)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4
            ),
            num_layers=num_layers
        )

    def forward(self, grid_features):
        """
        Args:
            grid_features: (B, hidden_dim, H, W)

        Returns:
            transformed: (B, hidden_dim, H, W)
        """
        # Patchify
        patches = self.patch_embed(grid_features)  # (B, N_patches, hidden_dim)

        # Transformer
        patches = patches.transpose(0, 1)  # (N_patches, B, hidden_dim)
        transformed = self.transformer(patches)
        transformed = transformed.transpose(0, 1)  # (B, N_patches, hidden_dim)

        # Unpatchify
        grid_out = self.unpatchify(transformed, grid_features.shape)

        return grid_out
```

**Novelty:** Transformer operates on graph-informed features (not raw pixels)

---

### 7. Physics-Constrained Decoder

**Goal:** Generate predictions that obey conservation laws

**Approach:** Physics-guided upsampling + PDE losses

```python
class PhysicsDecoder(nn.Module):
    def __init__(self, hidden_dim=128, output_steps=6):
        super().__init__()
        self.upsample = nn.ModuleList([
            UpBlock(hidden_dim, hidden_dim // 2),
            UpBlock(hidden_dim // 2, hidden_dim // 4),
            UpBlock(hidden_dim // 4, output_steps)
        ])

        # Physics parameters (learnable)
        self.advection_speed = nn.Parameter(torch.tensor(10.0))

    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim, H', W')

        Returns:
            predictions: (B, output_steps, 384, 384)
        """
        x = features
        for layer in self.upsample:
            x = layer(x)

        return x  # (B, output_steps, 384, 384)

    def physics_loss(self, pred, true, dt=5):
        """
        Enforce mass conservation and smoothness.

        Args:
            pred: (B, T, H, W) predictions
            true: (B, T, H, W) ground truth
            dt: time step in minutes

        Returns:
            physics_loss: scalar
        """
        # Mass conservation: ∂VIL/∂t + ∇·(v·VIL) ≈ 0
        dvil_dt = (pred[:, 1:] - pred[:, :-1]) / dt

        # Advection term (simplified: assume constant velocity)
        grad_x = pred[:, :-1, :, 1:] - pred[:, :-1, :, :-1]
        grad_y = pred[:, :-1, 1:, :] - pred[:, :-1, :-1, :]

        # Penalize large deviations from conservation
        conservation_error = dvil_dt + self.advection_speed * (grad_x.mean(-1) + grad_y.mean(-2))

        return conservation_error.abs().mean()
```

**Novelty:**
- Learnable physics parameters (advection speed)
- Differentiable PDE constraints
- Prevents physically impossible predictions

---

## Complete Architecture

```python
class StormGraphTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_gnn_layers=3, num_tf_layers=4):
        super().__init__()

        # 1. Multi-modal encoder
        self.encoder = MultiModalEncoder(hidden_dim)

        # 2. Storm detection
        self.detector = StormCellDetector()
        self.graph_builder = StormGraphBuilder()

        # 3. GNN
        self.gnn = StormGNN(hidden_dim, num_gnn_layers)

        # 4. Graph to grid
        self.graph_to_grid = GraphToGrid(grid_size=(96, 96), hidden_dim=hidden_dim)

        # 5. Transformer
        self.transformer = SpatioTemporalTransformer(hidden_dim, num_layers=num_tf_layers)

        # 6. Physics decoder
        self.decoder = PhysicsDecoder(hidden_dim, output_steps=6)

    def forward(self, vil, ir069, ir107, glm):
        # Encode
        features = self.encoder(vil, ir069, ir107, glm)  # (B, C, H', W')

        # Detect storms and build graph
        nodes, positions = self.detector(features, vil)

        batch_outputs = []
        for b in range(vil.size(0)):
            # Build graph for this sample
            edge_index, edge_attr = self.graph_builder.build_graph(positions[b])

            # GNN: model storm interactions
            updated_nodes = self.gnn(nodes[b], edge_index, edge_attr)

            # Convert back to grid
            grid = self.graph_to_grid(updated_nodes, positions[b])

            batch_outputs.append(grid)

        # Stack batch
        grid_features = torch.stack(batch_outputs)  # (B, C, H', W')

        # Transformer: global dependencies
        transformed = self.transformer(grid_features)

        # Decode with physics
        predictions = self.decoder(transformed)  # (B, 6, 384, 384)

        return predictions

    def compute_loss(self, pred, true, lambda_physics=0.1):
        # Standard MSE
        mse_loss = F.mse_loss(pred, true)

        # Physics constraint
        physics_loss = self.decoder.physics_loss(pred, true)

        # Total
        total = mse_loss + lambda_physics * physics_loss

        return total, {'mse': mse_loss, 'physics': physics_loss}
```

---

## Training Strategy

### Loss Function

```python
Total Loss = MSE + λ_physics * Physics + λ_extreme * Extreme
```

**Components:**
1. **MSE:** Pixel-wise accuracy
2. **Physics:** Conservation law violations
3. **Extreme:** Weighted loss for VIL > 181 (from Stage 4 insight!)

### Data Augmentation

- Random flips, rotations (preserve physics)
- Temporal shifts (different forecast horizons)
- Modality dropout (robustness to missing sensors)

### Training Schedule

```python
# Stage 1: Train encoder + decoder only (3 days)
# Stage 2: Add GNN, freeze encoder (2 days)
# Stage 3: Add Transformer, full end-to-end (3 days)
# Stage 4: Fine-tune with physics loss (2 days)
```

**Total:** ~10 days GPU training

---

## Evaluation Metrics

### Forecast Skill
- CSI@74, CSI@181, CSI@219 (per lead time)
- POD, SUCR, BIAS
- Fractions Skill Score (FSS)

### Physical Consistency
- Mass conservation error
- Advection alignment with optical flow
- Energy dissipation rate

### Interpretability
- Attention weight visualization (which storms matter?)
- Graph structure evolution (merging events)
- Physics parameter analysis

---

## Baselines to Compare

1. **Persistence** (copy last frame)
2. **Optical Flow** (advection-based)
3. **UNet2D** (our Stage 4)
4. **ConvLSTM** (our Stage 3)
5. **DGMR** (DeepMind's GAN-based, if we can replicate)
6. **MetNet-2** (Google's architecture, simplified version)

**Goal:** Show SGT outperforms all, especially on extreme events

---

## Novel Contributions

1. **First GNN-Transformer hybrid for weather nowcasting**
   - Prior work: Either CNNs OR Transformers, not hybrid with GNN

2. **Physics-informed graph construction**
   - Storm cells as nodes (physically meaningful)
   - Edge features encode advection

3. **Interpretable attention**
   - Can visualize which storms influence prediction
   - Useful for meteorologists

4. **Extreme event focus**
   - Leverages Stage 4 data scaling insight
   - Weighted loss for rare but important events

---

## Paper Outline

**Title:** "Physics-Informed Graph Neural Networks with Transformers for Severe Weather Nowcasting"

**Abstract:** (~200 words)
- Problem: Current models treat weather as continuous fields, miss discrete structure
- Solution: Storm-Graph Transformer (SGT) - hybrid GNN-Transformer with physics
- Results: X% improvement on extreme events, physically consistent, interpretable

**1. Introduction**
- Motivation: Severe weather forecasting challenges
- Limitations of current deep learning approaches
- Our contributions (3-4 bullet points)

**2. Related Work**
- CNN-based nowcasting (UNet, ConvLSTM)
- Transformer architectures (MetNet-2, Earthformer)
- GNNs in weather/climate (rare, cite what exists)
- Physics-informed neural networks

**3. Method**
- 3.1 Problem formulation
- 3.2 Multi-modal encoder
- 3.3 Storm graph construction
- 3.4 GNN module
- 3.5 Transformer module
- 3.6 Physics-constrained decoder

**4. Experiments**
- 4.1 Dataset (SEVIR, 541 events, 4 modalities)
- 4.2 Baselines
- 4.3 Metrics
- 4.4 Results
  - Overall performance
  - Per-lead-time analysis
  - Extreme event performance
  - Ablations (GNN-only, Transformer-only, no physics)

**5. Analysis**
- 5.1 Attention visualization
- 5.2 Graph structure evolution
- 5.3 Physics consistency
- 5.4 Failure cases

**6. Conclusion**
- Summary
- Limitations
- Future work

**References:** (~30-40 papers)

---

## Timeline (3 Weeks)

**Week 1 (Oct 10-17):**
- Day 1-2: Implement multimodal data loader
- Day 3-4: Implement encoder + storm detection
- Day 5-6: Implement GNN module
- Day 7: Integration testing

**Week 2 (Oct 17-24):**
- Day 1-2: Implement Transformer module
- Day 3-4: Implement physics decoder
- Day 5-7: Full pipeline training (Stage 1-2)

**Week 3 (Oct 24-31):**
- Day 1-3: Full training (Stage 3-4)
- Day 4-5: Baseline comparisons + ablations
- Day 6-7: Experiments + analysis

**Week 4 (Oct 31-Nov 7):**
- Day 1-4: Paper writing
- Day 5-7: Figures, proofreading, ArXiv submission

---

## Success Criteria

**Minimum Viable:**
- ✅ Architecture implements as designed
- ✅ Trains stably, no collapse
- ✅ Matches or beats UNet2D baseline (CSI@74 ≥ 0.82)
- ✅ Shows some interpretability (attention maps)

**Target:**
- ✅ Beats all baselines on extreme events (CSI@181 > 0.50)
- ✅ Physics loss reduces conservation error by 30%+
- ✅ Attention reveals meaningful storm interactions
- ✅ Publication-quality writing + figures

**Stretch:**
- ✅ SOTA on SEVIR benchmark
- ✅ Qualitative validation by meteorologist
- ✅ Real-time inference capability (<1s per forecast)

---

*This is our flagship paper. Let's make it count!*

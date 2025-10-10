
# Architecture Ladder

1. **Baselines**
   - Persistence (copy last frame)
   - Optical flow advection (e.g., Horn–Schunck-like, rainymotion-style)
2. **U-Net 2D**
   - Encode stacked input slices as channels; decode future frames.
3. **ConvLSTM / ConvGRU**
   - Spatiotemporal recurrence; encoder–decoder; skip connections.
4. **Attention U-Net**
   - Channel & spatial attention; cross-time attention.
5. **Spatiotemporal Transformer**
   - Patch tokens over (H,W,T); relative positional encodings; windowed attention.
6. **Generative Texture**
   - cGAN (PatchGAN) or diffusion prior; supervised content loss anchor.
7. **Fusion Models**
   - Multi-branch encoders per modality; alignment & learned warping; late fusion heads.
8. **Novel Directions**
   - Flow-guided attention, dynamic convolution, neural operators for advection, implicit neural fields for time.

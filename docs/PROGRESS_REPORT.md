# StormFusion-SEVIR: Progress Report & Context

**Date:** October 11, 2025
**Status:** Active Development - Paper 1 Implementation Phase
**Timeline:** 6 weeks to complete 3 publication-grade papers (deadline: Thanksgiving, Nov 27)

---

## Project Overview

### Research Goals
This is **NOT a tutorial project** - it's a research effort aimed at **3 publication-grade papers** for both academic and industry impact.

**Core Mission:**
1. Advance state-of-the-art in severe weather nowcasting
2. Introduce novel architectures (GNN-Transformer-Physics hybrid)
3. Enhance existing datasets (SEVIR → SEVIR++)
4. Enable probabilistic forecasting at city-block resolution (1-2km)

**Target Venues:**
- ArXiv preprints (citable, industry-relevant)
- NeurIPS/ICML workshops
- Domain journals (Weather & Forecasting, IEEE TGRS)

---

## Current State of Project

### What We've Accomplished (Stages 0-4)

#### ✅ Stage 0-3: Baseline Pipeline (COMPLETE)
- Environment setup, data loading, basic metrics
- U-Net baseline: CSI@74 = 0.68 (tiny dataset)
- ConvLSTM temporal: CSI@74 = 0.73 (tiny dataset)
- **Key insight:** Pipeline works, ready for scaling

#### ✅ Stage 4: Data Scaling Breakthrough (COMPLETE) 🎉
**Problem Identified:**
- 60-event dataset: Model catastrophically fails on extreme weather
- CSI@181 (extreme) = 0.16, CSI@219 (hail) = 0.08 ❌
- Systematic underestimation of high-intensity events

**Root Cause Found:**
- Data scarcity, NOT model architecture limitation
- 60 events insufficient for rare extreme event patterns

**Solution Validated:**
- Scaled to ALL 541 SEVIR events (432 train / 109 val)
- Results: CSI@181 = 0.50 (+212%), CSI@219 = 0.33 (+318%) ✅
- **Key insight:** More data > fancier loss functions

**Documented:**
- `docs/STAGE4_BREAKTHROUGH.md` - Full analysis
- `docs/STAGE4_NEXT_STEPS.md` - Implementation details
- `docs/WHY_PERCEPTUAL_LOSS_MATTERS.md` - Research context

#### ✅ Stage 5: Multi-Step Forecasting (DESIGNED)
- Notebook created: `Stage05_Multi_Step_Forecasting.ipynb`
- 12 input → 6 output frames (5-30 min predictions)
- Per-lead-time metrics tracking
- **Status:** Ready to run, but deprioritized for Paper 1 focus

---

## Three-Paper Strategy

### Paper 1: "Physics-Informed Graph-Transformer for Severe Weather Nowcasting"
**Status:** 🔥 ACTIVE - Week 1 Implementation
**Timeline:** Oct 10-31 (3 weeks development) + Oct 31-Nov 7 (writing)
**Target:** ArXiv + NeurIPS workshop

**Core Innovation:**
- Hybrid GNN-Transformer-Physics architecture
- Treats storms as discrete graph nodes (not continuous CNN fields)
- Physics constraints (conservation laws, advection)
- Interpretable attention (which storms influence predictions?)

**Novel Contributions:**
1. First GNN-Transformer hybrid for weather nowcasting
2. Physics-informed graph construction (storm cells as nodes)
3. Extreme event focus (leverages Stage 4 data scaling insight)
4. Interpretable attention mechanisms

**What Makes It Novel:**
- Prior work: CNN-only (UNet, ConvLSTM) OR Transformer-only (MetNet-2)
- We combine: GNN (discrete storm structure) + Transformer (global context) + Physics (constraints)
- Graph explicitly models storm-storm interactions (merging, splitting, advection)

**Completed:**
- ✅ Architecture design: `docs/PAPER1_ARCHITECTURE.md`
- ✅ Multimodal data loader: `stormfusion/data/sevir_multimodal.py`
- ✅ Colab notebook: `notebooks/colab/Paper1_StormGraphTransformer.ipynb`

**In Progress:**
- ⏳ Verify SEVIR modalities (user testing in Colab now)
- ⏳ Storm cell detection module
- ⏳ GNN module implementation
- ⏳ Transformer module
- ⏳ Physics decoder

**Timeline:**
- Week 1 (Oct 10-17): Core modules (GNN, Transformer, Physics)
- Week 2 (Oct 17-24): Training pipeline + initial experiments
- Week 3 (Oct 24-31): Full experiments + baselines + ablations
- Week 4 (Oct 31-Nov 7): Paper writing + figures

---

### Paper 2: "SEVIR++: Enhanced Multimodal Storm Dataset"
**Status:** 📋 PLANNED - Starts Week 2
**Timeline:** Oct 17-Nov 7
**Target:** ArXiv + domain journal

**Core Innovation:**
- Expand SEVIR's 3 ABI channels → 13+ GOES-16 channels
- Engineer GLM features (flash density, growth rate, clustering)
- Derive physical variables (optical flow, gradients, CI indicators)

**Dataset Enhancements:**
- All visible bands (0.47, 0.64, 0.86 μm)
- Near-IR (1.6, 2.2 μm)
- Water vapor channels (6.2, 6.9, 7.3 μm)
- All IR bands (8.4-13.3 μm)
- Derived physics: convective initiation, vorticity proxies

**Novel Contributions:**
1. Most comprehensive multimodal storm dataset
2. Engineered physical features (beyond raw satellite data)
3. Benchmark tasks on enhanced modalities
4. Publicly released for community

**Status:**
- Need to assess: Can we download/process extra GOES-16 data in time?
- Fallback: Enhanced feature engineering on existing SEVIR (still publishable)

---

### Paper 3: "Probabilistic Storm Nowcasting with Spatially-Granular Uncertainty"
**Status:** 📋 PLANNED - Starts Week 4
**Timeline:** Oct 31-Nov 21
**Target:** ArXiv + workshop

**Core Innovation:**
- Conditional diffusion model for ensemble generation
- Trained on SEVIR++ (Paper 2's dataset)
- Maintains spatial granularity (perceptual loss foundation from Stage 4)
- Calibrated probabilistic outputs

**Novel Contributions:**
1. First diffusion model on multimodal weather data
2. Cross-modal conditioning in denoising process
3. City-block level uncertainty (1-2km resolution)
4. Combines Paper 1 architecture + Paper 2 dataset

**Dependencies:**
- Requires Paper 1 architecture working
- Ideally uses Paper 2 enhanced dataset
- Can fallback to GAN if diffusion too slow

---

## Critical Technical Details

### Data

**SEVIR Dataset:**
- 541 events total
- Split: 432 train / 109 val (80/20, seed=42)
- 4 default modalities: VIL, IR069, IR107, GLM
- Resolution: 384×384 @ 1km
- Temporal: 49 frames @ 5min intervals
- Location: `/content/drive/MyDrive/SEVIR_Data` (Colab)

**Key Files:**
- Catalog: `SEVIR_CATALOG.csv`
- Event IDs: `all_train_ids.txt`, `all_val_ids.txt`
- Data root: `data/sevir/` (subdirs: vil/, ir069/, ir107/, lght/)

**Normalization Stats (Paper 1):**
```python
NORM_STATS = {
    'vil': {'mean': 0.089, 'std': 0.178},
    'ir069': {'mean': 0.481, 'std': 0.156},
    'ir107': {'mean': 0.524, 'std': 0.130},
    'lght': {'mean': 0.003, 'std': 0.028}
}
```

---

### Stage 4 Breakthrough Details (CRITICAL CONTEXT)

**Why This Matters:**
This discovery fundamentally shapes all three papers.

**The Problem:**
Models trained on 60 events showed catastrophic failure on extreme weather:
- Moderate events (VIP 74): CSI = 0.70 ✅
- Severe (VIP 160): CSI = 0.27 ❌
- Extreme (VIP 181): CSI = 0.16 ❌
- Hail (VIP 219): CSI = 0.08 ❌ (almost useless)

**Hypothesis:**
Was this a model architecture problem or data problem?

**Test:**
Trained same architecture (UNet2D) on ALL 541 events instead of 60.

**Results:**
- Moderate (VIP 74): CSI = 0.82 (improved!)
- Extreme (VIP 181): CSI = 0.50 (+212% improvement!) 🎉
- Hail (VIP 219): CSI = 0.33 (+318% improvement!) 🎉

**Conclusions:**
1. Data scarcity was the bottleneck, not model capacity
2. Rare events need proportionally more examples (~100× rule)
3. Simple solutions (more data) often beat complex ones (fancy losses)
4. Pure MSE with sufficient data > MSE + Perceptual Loss

**Impact on Papers:**
- Paper 1: Use all 541 events, focus on extreme event performance
- Paper 2: Dataset contribution more valuable (quantity matters!)
- Paper 3: Can use simpler baselines, data is foundation

**Key Files:**
- `docs/STAGE4_BREAKTHROUGH.md` - Full analysis
- `notebooks/colab/Stage04_ALL_EVENTS_Extreme_Fix.ipynb` - Experiments
- Charts: `stage04charts.png` - Visualizations showing problem

---

## Paper 1: Architecture Details

### Storm-Graph Transformer (SGT)

**High-Level Pipeline:**
```
Input: [VIL, IR069, IR107, GLM] × 12 timesteps
  ↓
1. Multi-Modal Encoder (separate CNN per modality)
  ↓
2. Storm Cell Detection (identify discrete entities)
  ↓
3. Graph Construction (cells = nodes, proximity = edges)
  ↓
4. GNN Layers (model storm interactions)
  ↓
5. Graph → Grid Projection (Gaussian splatting)
  ↓
6. Transformer Layers (global spatiotemporal attention)
  ↓
7. Physics-Constrained Decoder (conservation laws)
  ↓
Output: VIL × 6 timesteps (5-30 min predictions)
```

**Key Modules:**

1. **MultiModalEncoder:**
   - Per-modality ResNet encoders
   - Fusion layer combines features
   - Output: (B, 128, 96, 96) encoded features

2. **StormCellDetector:**
   - Watershed segmentation on VIL peaks
   - Extracts features at storm locations
   - Dynamic: N_storms varies per sample
   - Output: Node features (N, 128), positions (N, 2)

3. **StormGraphBuilder:**
   - k-NN graph (k=8 neighbors)
   - Edge features: distance + direction vector
   - Encodes spatial relationships
   - Output: edge_index (2, E), edge_attr (E, 3)

4. **StormGNN:**
   - Graph Attention Network (GAT)
   - 3 layers, 4 attention heads
   - Learns storm-storm interactions
   - Output: Updated node features (N, 128)

5. **GraphToGrid:**
   - Projects graph back to spatial grid
   - Gaussian splatting (learnable)
   - Output: Grid features (B, 128, 96, 96)

6. **SpatioTemporalTransformer:**
   - Vision Transformer on patches
   - 4 layers, 8 attention heads
   - Global context modeling
   - Output: Transformed grid (B, 128, 96, 96)

7. **PhysicsDecoder:**
   - Upsampling to (B, 6, 384, 384)
   - Physics loss: mass conservation
   - Learnable advection parameters
   - Output: 6-frame predictions

**Loss Function:**
```python
Total = MSE + λ_physics * Physics + λ_extreme * Extreme
```
- MSE: Pixel-wise accuracy
- Physics: Conservation law violations
- Extreme: Weighted loss for VIP > 181 (Stage 4 insight!)

**Training Strategy:**
1. Stage 1: Encoder + Decoder only (3 days)
2. Stage 2: Add GNN, freeze encoder (2 days)
3. Stage 3: Add Transformer, end-to-end (3 days)
4. Stage 4: Fine-tune with physics (2 days)

**Baselines to Beat:**
- Persistence (copy last frame)
- Optical Flow (advection)
- UNet2D (Stage 4: CSI@74 = 0.82)
- ConvLSTM (Stage 3: CSI@74 = 0.73)
- DGMR (DeepMind GAN, if replicable)
- MetNet-2 (Google, simplified)

---

## Implementation Status

### Repository Structure
```
stormfusion-sevir/
├── docs/
│   ├── PAPER1_ARCHITECTURE.md         ✅ Complete
│   ├── STAGE4_BREAKTHROUGH.md         ✅ Complete
│   ├── EXPERIMENT_PLAN.md             ✅ Reference
│   ├── ARCHITECTURE_LADDER.md         ✅ Reference
│   ├── DATA_ENHANCEMENT.md            ✅ Reference
│   └── WHY_PERCEPTUAL_LOSS_MATTERS.md ✅ Reference
│
├── stormfusion/
│   └── data/
│       └── sevir_multimodal.py        ✅ Complete
│   └── models/
│       └── sgt/                       ⏳ In Progress
│           ├── __init__.py            ⏳ Pending
│           ├── encoder.py             ⏳ Pending
│           ├── detector.py            ⏳ Pending
│           ├── gnn.py                 ⏳ Pending
│           ├── transformer.py         ⏳ Pending
│           └── decoder.py             ⏳ Pending
│
├── notebooks/colab/
│   ├── Paper1_StormGraphTransformer.ipynb  ✅ Complete (testing)
│   ├── Stage04_ALL_EVENTS_Extreme_Fix.ipynb ✅ Complete
│   └── Stage05_Multi_Step_Forecasting.ipynb ✅ Complete (deprioritized)
│
├── scripts/
│   ├── verify_sevir_modalities.py     ✅ Complete
│   └── analyze_sevir_extremes.py      ✅ Complete
│
└── data/samples/
    ├── all_train_ids.txt              ✅ 432 events
    └── all_val_ids.txt                ✅ 109 events
```

---

## Current Blocker & Next Steps

### BLOCKER: Data Verification
**Status:** User testing in Colab (in progress)

**What we need to know:**
Which SEVIR modalities are available?
- ✅ VIL (confirmed - has been using this)
- ❓ IR069 (water vapor)
- ❓ IR107 (IR window)
- ❓ GLM (lightning)

**Notebook to run:**
`Paper1_StormGraphTransformer.ipynb` - Section 1 (Data Verification)

**Possible Outcomes:**

1. **All 4 modalities available** → Full steam ahead on GNN-Transformer
   - Implement all modules as designed
   - Flagship paper with multimodal fusion

2. **2-3 modalities available** → Adapt architecture
   - Still strong paper
   - Adjust encoder to available modalities
   - May need to download missing ones

3. **VIL only** → Pivot strategy
   - Focus on Stage 4 data scaling paper first
   - Or implement GNN on VIL-only (still novel)
   - Download other modalities in parallel

### Next Implementation Steps (Once Data Verified)

**Priority 1: Storm Cell Detection (Day 2)**
- Implement peak detection on VIL
- Extract features at storm locations
- Test on sample images
- File: `stormfusion/models/sgt/detector.py`

**Priority 2: Graph Construction (Day 2-3)**
- k-NN graph builder
- Edge feature extraction (distance, direction)
- Test graph connectivity
- File: `stormfusion/models/sgt/gnn.py` (part 1)

**Priority 3: GNN Module (Day 3-4)**
- Graph Attention Network implementation
- Message passing
- Test on toy graphs first
- File: `stormfusion/models/sgt/gnn.py` (part 2)

**Priority 4: Integration (Day 5-7)**
- Connect all modules
- End-to-end forward pass
- Debug shape mismatches
- Small-scale training test

---

## Key Decisions & Rationale

### Why This Changed from Original Plan

**Original Plan:**
- Stage 5: Multi-step forecasting (extend Stage 4)
- Linear progression through stages
- Single-modality (VIL only)

**Why We Pivoted:**
1. **Timeline pressure:** 6 weeks to 3 papers, need parallelization
2. **Novelty requirement:** Multi-step VIL-only is incremental, not novel enough
3. **Architecture innovation:** GNN-Transformer is truly novel contribution
4. **Multimodal necessity:** All top papers use multiple modalities
5. **Industry relevance:** Hybrid physics-ML is current frontier

**This is the RIGHT decision because:**
- Paper 1 (GNN-Transformer) is flagship, high impact
- Can leverage Stage 4 insights (extreme events)
- Multimodal data makes results stronger
- Architecture novelty carries entire paper

### Why Stage 4 Matters So Much

The data scaling breakthrough changes everything:
1. Validates that we can solve extreme events (core problem in field)
2. Shows more data > complex architectures (practical insight)
3. Establishes 541-event dataset as minimum viable
4. Provides strong baseline to beat (CSI@74 = 0.82)

Without Stage 4, Paper 1 would struggle with extreme events too. Now we know the solution and can focus on architecture novelty.

---

## Compute Requirements

**Estimated GPU Hours:**

Paper 1 Training:
- Module development/testing: ~10 GPU hours
- Baseline experiments: ~20 GPU hours (5 models × 4 hours each)
- Full SGT training: ~30 GPU hours (progressive training)
- Ablations: ~20 GPU hours
- **Total:** ~80 GPU hours

**Colab Strategy:**
- Free tier: ~12 hours/day (with disconnects)
- Need: ~8 days of training
- Parallel: Run multiple experiments per day
- **Feasible but tight**

**Risk Mitigation:**
- Implement checkpointing early
- Test on small subset first (10 events)
- Scale up gradually
- Consider Colab Pro if needed (~$10/month)

---

## Success Criteria

### Paper 1 Minimum Viable:
- ✅ Architecture implements as designed
- ✅ Trains stably, converges
- ✅ Matches or beats UNet baseline (CSI@74 ≥ 0.82)
- ✅ Shows interpretability (attention visualizations)
- ✅ Physics loss provides measurable benefit

### Paper 1 Target:
- ✅ Beats all baselines on extreme events (CSI@181 > 0.50)
- ✅ Physics loss reduces conservation error >30%
- ✅ Attention reveals meaningful storm interactions
- ✅ Ablations show each component helps
- ✅ Publication-quality writing + figures

### Paper 1 Stretch:
- ✅ State-of-the-art on SEVIR benchmark
- ✅ Real-time inference (<1s per forecast)
- ✅ Qualitative validation by meteorologist

---

## Red Flags & Contingencies

### Potential Issues:

1. **Missing SEVIR modalities**
   - Contingency: Download from MIT SEVIR or AWS S3
   - Timeline impact: +2 days
   - Fallback: Implement on VIL-only, adapt architecture

2. **GNN doesn't converge**
   - Contingency: Simplify to message passing (no attention)
   - Fallback: Skip GNN, pure Transformer (still novel with physics)

3. **Training too slow**
   - Contingency: Reduce model size, fewer layers
   - Get Colab Pro ($10/month for faster GPUs)
   - Parallelize experiments better

4. **Physics loss hurts performance**
   - Contingency: Make it optional/weighted lower
   - Still discuss in paper as attempted approach

5. **Can't beat baselines**
   - Contingency: Paper becomes "analysis of why GNN-Transformer doesn't help"
   - Still publishable (negative results matter)
   - Pivot to Paper 2 (dataset) as primary contribution

### Timeline Risks:

- Week 1 delay → compress Week 2 (fewer ablations)
- Week 2 delay → extend to Week 4, Paper 2 becomes brief
- Week 3 delay → Paper 3 becomes position paper / future work

**Worst case:** 1 strong paper + 2 workshop/position papers (still good outcome)

---

## Context for Future Claude Sessions

### If This Chat Fails, Next Claude Should Know:

**Where we are:**
- Day 1 of Week 1 for Paper 1
- Architecture designed, data loader ready
- User is testing multimodal data in Colab NOW
- Waiting for confirmation of which SEVIR modalities available

**What to do immediately:**
1. Ask user: "What SEVIR modalities do you have? (VIL, IR069, IR107, GLM)"
2. Based on answer, prioritize next module implementation
3. Start with storm detection module (most critical)
4. Keep parallel track on Paper 2 planning (dataset enhancement)

**Key principles:**
- **Speed over perfection** (6-week deadline)
- **Novel over incremental** (needs to be publication-grade)
- **Test early, test often** (Colab has limits)
- **Document as you go** (for paper writing later)

**User's mindset:**
- Ambitious (3 papers in 6 weeks)
- Pragmatic (wants industry + academic impact)
- Experienced researcher (knows what publications need)
- Time-constrained (Thanksgiving deadline firm)

**Communication style:**
- Direct, technical, no fluff
- Acknowledge tradeoffs openly
- Provide contingencies for risks
- Focus on actionable next steps

---

## Critical Files to Reference

**Must read before continuing:**
1. `docs/PAPER1_ARCHITECTURE.md` - Complete architecture spec
2. `docs/STAGE4_BREAKTHROUGH.md` - Why data scaling matters
3. `stormfusion/data/sevir_multimodal.py` - Current data loader
4. This file: `docs/PROGRESS_REPORT.md` - You are here

**Check these for context:**
- `docs/EXPERIMENT_PLAN.md` - Original staged plan
- `docs/WHY_PERCEPTUAL_LOSS_MATTERS.md` - Loss function insights
- README.md - Project overview and current progress

**Notebooks to understand:**
- `Paper1_StormGraphTransformer.ipynb` - Main training notebook
- `Stage04_ALL_EVENTS_Extreme_Fix.ipynb` - Data breakthrough results

---

## Quick Reference: Key Numbers

**Dataset:**
- Total events: 541 (432 train, 109 val)
- Image size: 384×384 @ 1km
- Time steps: 49 frames @ 5min
- Input: 12 frames (0-55 min history)
- Output: 6 frames (5-30 min predictions)

**Stage 4 Results (Baseline to Beat):**
- CSI@74 (moderate): 0.82
- CSI@181 (extreme): 0.50
- CSI@219 (hail): 0.33
- LPIPS (sharpness): 0.137

**VIP Thresholds:**
- 16: Light precipitation
- 74: Moderate (main metric)
- 133: Heavy
- 160: Severe
- 181: Extreme
- 219: Hail

**Model Config:**
- Hidden dim: 128
- GNN layers: 3
- Transformer layers: 4
- Attention heads: 8
- Batch size: 4
- Learning rate: 1e-4

---

## Team & Resources

**Solo researcher:** User (Haider)
**AI Assistant:** Claude (context resets possible)
**Compute:** Google Colab (free tier, limited)
**Data:** SEVIR dataset (MIT, on Google Drive)
**Code:** GitHub repo (syedhaliz/stormfusion-sevir)

**No collaborators mentioned** → Full autonomy on decisions
**No advisor constraints** → Can move fast
**Timeline is FIRM** → Thanksgiving deadline non-negotiable

---

## Mindset & Philosophy

**From user:** "Can't be shy on the data, all SEVIR all channels, all modalities"

**From Stage 4:** "More data > fancier loss functions"

**From 3-paper strategy:** "Novel over incremental"

**Key insight:** This is about advancing the field, not just completing assignments.

---

## Final Note for Future Claude

If you're reading this, the original conversation ended. Here's what matters:

1. **Check git log** - See latest commits for progress
2. **Ask user for data status** - Do they have all 4 SEVIR modalities?
3. **Implement next module** - Storm detection is most critical
4. **Test frequently** - Colab can disconnect anytime
5. **Document progress** - Update this file as needed

**Don't reinvent the wheel** - Architecture is designed, just implement it.

**Don't overthink** - User wants action, not more planning.

**Trust Stage 4 insights** - Data scaling works, use 541 events.

**Remember the goal** - 3 papers by Nov 27. Clock is ticking.

Good luck! 🚀

---

*Last updated: October 11, 2025, 2:00 PM*
*Current phase: Paper 1 Week 1 Day 1 - Data verification*
*Next milestone: Storm detection module implementation*

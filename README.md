# 🚗 SATT — Social-Attention Trajectory Transformer

**Intent & Trajectory Prediction for L4 Urban Autonomous Driving**

A pure-PyTorch implementation of a Transformer-based multi-modal trajectory prediction model trained on the nuScenes dataset. Predicts the **3 most likely future paths** (3 seconds) of pedestrians and cyclists based on 2 seconds of past motion and social context.

---

## 📊 Results

| Metric | Score |
|---|---|
| **Total minADE₃** | **0.213 m** |
| **Total minFDE₃** | **0.389 m** |
| Pedestrian minADE₃ | 0.215 m (n=837) |
| Bicycle minADE₃ | 0.172 m (n=35) |

## 🏗️ Architecture

```
Input (x, y, dx, dy, heading, class, moving)
        │
        ▼
┌─────────────────────┐
│  Temporal Encoder    │  ← 2-layer Transformer + Positional Encoding
│  (5 history steps)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────┐
│  Social Attention    │◄────│ Neighbor Encoder  │ ← Dedicated 2-layer MLP
│  (Multi-Head + Skip) │     │ (20 closest agents)│
└────────┬────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────────┐
│  Multi-Modal Decoder │  ← K=3 modes, 3-layer MLP + Residual
│  + Confidence Head   │    LayerNorm-stabilized logits
└────────┬────────────┘
         │
         ▼
   K=3 predicted trajectories + confidence scores
```

## 🔧 Key Technical Features

- **Coordinate Heading Invariance**: All trajectories rotated to agent-local +Y frame
- **Winner-Takes-All Loss**: Only backpropagates through the closest mode
- **Hinge Diversity Loss**: Prevents mode collapse by penalizing modes < 0.5m apart
- **Euclidean Social Sorting**: Nearest 20 neighbors selected by distance
- **Gaussian Noise Augmentation**: Applied to history only (not ground truth)
- **Early Stopping**: Patience-based (stops when validation plateaus)
- **LR Warmup + Cosine Decay**: Stabilized Transformer training

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Place the nuScenes mini dataset
# Download from: https://www.nuscenes.org/nuscenes#download
# Extract to: v1.0-mini/v1.0-mini/

# 3. Train (~3 min on M4 Mac)
python train.py

# 4. Evaluate + Visualize
python evaluate.py

# 5. Quick inference demo
python predict.py
```

## 📁 Project Structure

```
├── data/
│   └── dataset.py              # Zero-dependency nuScenes JSON parser
├── model/
│   ├── encoder.py              # Transformer temporal encoder
│   ├── social_attention.py     # Multi-head attention + residual skip
│   ├── decoder.py              # K=3 multi-modal decoder + LayerNorm
│   └── trajectory_predictor.py # Full SATT assembly
├── utils/
│   └── metrics.py              # minADE/minFDE computation
├── train.py                    # Training loop (WTA + diversity + warmup)
├── evaluate.py                 # Dual-class eval + TTA + BEV plots
├── predict.py                  # CLI inference demo
└── requirements.txt            # Dependencies
```

## 📦 Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, tqdm

## 📈 Training Progression

| Version | minADE₃ | Key Change |
|---|---|---|
| v1 | 0.293m | Base SATT architecture |
| v2 | 0.284m | Diversity loss + grad clipping |
| v3 | 0.267m | Heading invariance + neighbor sorting |
| **v4** | **0.213m** | 10-fix audit (noise bug, deep decoder, residuals) |

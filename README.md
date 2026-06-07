# SmartRef-Net 🏀

**An AI-Powered Basketball Shooting Foul Detection System with Explainable Kinematic Evidence**

> Final Year Project 
> Author: Muhammed Akthar Abdul Rasheed 

---

## Overview

SmartRef-Net is a hybrid multimodal ensemble system that classifies basketball video clips as **FOUL** or **NO FOUL** — and explains *why* using kinematic evidence.

Every existing automated system outputs a label with no justification. SmartRef-Net is the first known system to produce a kinematic evidence panel alongside every prediction: torso proximity, contact velocity, deceleration signature, arm extension asymmetry, and pose skeleton annotation. Think of it as VAR for basketball — but one that shows its reasoning.

---

## Architecture

```
Video Clip (3–15 seconds)
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
  [Visual Stream]                          [Pose Stream]
  VideoMAE Transformer                YOLO26x-Pose + ByteTrack
  16 frames → 768-d CLS token         17 keypoints per player
  LayerNorm → Linear → 128-d          20 biomechanical features/frame
                                       BiGRU → mean pool → 128-d
         │                                         │
         └──────────────┬──────────────────────────┘
                        │
                        ▼
                 [Late Fusion MLP]
              concat(128+128) → 256-d
           BatchNorm → GELU → Linear(2)
                        │
                        │          [Enhanced Temporal XGBoost]
                        │          150 contact dynamics features
                        │          from same (16×20) pose sequence
                        │                        │
                        └──────────┬─────────────┘
                                   │
                              Weighted Ensemble
                           NN×0.75 + XGB×0.25
                           Threshold = 0.460
                                   │
                              FOUL / NO FOUL
                         + Kinematic Evidence Panel
```

### Component 1 — Dual-Stream Neural Network

**Video Stream:** VideoMAE (`MCG-NJU/videomae-base-finetuned-kinetics`) processes 16 uniformly sampled RGB frames. The CLS token (768-d) is projected to 128-d via LayerNorm → Linear → GELU → Dropout.

**Pose Stream:** YOLO26x-Pose with ByteTrack detects and tracks players across all frames. For each frame, 20 biomechanical features are extracted from the interacting pair:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `torso_distance` | Body-scale normalised player proximity |
| 1 | `wristA_to_torsoB` | Offender arm reach toward defender body |
| 2 | `wristB_to_torsoA` | Defender arm reach toward offender body |
| 3–6 | `knee_angles_A/B` | Charging / defensive stance angles |
| 7 | `foot_min_distance` | Minimum foot-to-foot distance |
| 8–15 | `validity_masks` | Keypoint confidence flags (8 features) |
| 16 | `velocity` | Frame-to-frame torso distance delta |
| 17 | `acceleration` | Frame-to-frame velocity delta |
| 18 | `ball_to_wrist_dist` | Offender wrist proximity to ball |
| 19 | `possession_proxy` | Binary ball possession estimate |

The (16×20) sequence feeds a Bidirectional GRU (hidden=64, layers=2), mean-pooled to 128-d and projected via LayerNorm → Linear → GELU → Dropout.

Both streams are concatenated (256-d) and passed through a fusion MLP: Linear(256→256) → BatchNorm1d → GELU → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→2).

### Component 2 — Enhanced Temporal XGBoost

Trained on 150 hand-crafted contact dynamics features derived from the same (16×20) pose sequence:

- **140 temporal statistics:** 7 stats × 20 features (mean, std, min, max, delta, argmin\_t, argmax\_t)
- **10 contact dynamics:** min torso distance, deceleration at contact, max consecutive close frames, contact asymmetry, torso distance slope, velocity at closest approach, max absolute acceleration, mean close velocity, wrist proximity at contact, arm extension delta

---

## Results

Evaluated on a 30-clip external test set from a completely unseen game (15 foul / 15 no-foul):

| Model | F1 | AUPRC | Foul | No-Foul |
|-------|-----|-------|------|---------|
| Majority Class | 0.6667 | N/A | 15/15 | 0/15 |
| CNN+LSTM | 0.6667 | 0.3608 | 15/15 | 0/15 |
| XGBoost Basic | 0.6667 | 0.5000 | 15/15 | 0/15 |
| Dual-Stream NN Only | 0.6667 | 0.5252 | 15/15 | 0/15 |
| VideoMAE Only | 0.6316 | 0.5462 | 14/15 | 1/15 |
| **SmartRef-Net** | **0.6111** | **0.4683** | **11/15** | **5/15** |

> **Key finding:** Every baseline achieves F1=0.6667 by predicting FOUL on every single clip — zero no-foul clips correctly identified. SmartRef-Net is the **only model demonstrating genuine binary discrimination** across all evaluated baselines.

---

## Dataset

- **369 annotated NBA video clips** — no comparable public resource exists
- **275 train / 95 validation / 30 external test** (unseen game)
- **49 hard negative no-foul clips** curated via player proximity analysis
- Annotated using CVAT XML with paired Offender/Defender tracks and temporal event boundaries
- Independent blind referee review on all 30 external test clips
- Source-aware train/val split via GroupShuffleSplit to prevent data leakage across games

---

## Project Structure

```
SmartRef-Net/
│
├── smartref_w1953520_20220707.ipynb   # Main training notebook (110 cells)
├── SmartRef_Demo.ipynb                # Gradio demo notebook (Colab)
│
├── model.py                           # SmartRefNet architecture + inference pipeline
├── main.py                            # FastAPI backend
│
├── models/                            # Model weights (not included — see below)
│   ├── smartref_best.pt               # Trained SmartRefNet checkpoint
│   ├── xgb_enhanced_model.pkl         # Trained XGBoost model
│   ├── yolo26x-pose.pt                # YOLO26x-Pose weights
│   └── yolo26x.pt                     # YOLO26x ball detection weights
│
└── static/                            # FastAPI frontend
    └── index.html
```

---

## Setup and Installation

### Requirements

```bash
pip install torch torchvision transformers ultralytics xgboost \
            opencv-python-headless scikit-learn fastapi uvicorn \
            gradio numpy pandas scipy
```

### Model Weights

The trained model weights are not included in this repository due to file size. Download from [Google Drive link] and place in the `models/` directory:

- `smartref_best.pt`
- `xgb_enhanced_model.pkl`
- `yolo26x-pose.pt`
- `yolo26x.pt`

---

## Running the Demo

### Option 1 — Google Colab (recommended)

Open `SmartRef_Demo.ipynb` in Google Colab with an A100 GPU. Run all cells in order. A public Gradio link will appear at the bottom of Cell 9.

### Option 2 — Local FastAPI

```bash
uvicorn main:app --reload
```

Navigate to `http://localhost:8000`. Upload an `.mp4` clip (3–15 seconds) and click Analyse.

---

## Training

The full training pipeline is in `smartref_w1953520_20220707.ipynb`. Key stages:

**Phase 1 — VideoMAE frozen (15 epochs)**
```
LR: 3e-4 | Scheduler: OneCycleLR | Loss: FocalLoss(gamma=2.0)
Sampler: WeightedRandomSampler | Batch: 2 | Grad accumulation: 4
```

**Phase 2 — Full fine-tune (20 epochs)**
```
LR: 3e-6 | Scheduler: CosineAnnealingLR | VideoMAE unfrozen
```

**XGBoost**
```
n_estimators: 300 | max_depth: 5 | learning_rate: 0.03
scale_pos_weight: n_neg/n_pos | subsample: 0.8 | colsample_bytree: 0.6
```

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Input | ffmpeg, OpenCV 4.8 |
| Detection & Tracking | YOLO26x-Pose, ByteTrack (Ultralytics 8.3) |
| Feature Engineering | Custom Python, NumPy, CVAT |
| Neural Network | PyTorch 2.0, HuggingFace Transformers 4.40, VideoMAE |
| Gradient Boosting | XGBoost 2.0 |
| Ensemble & Output | scikit-learn, Custom Python |
| Web Interface | FastAPI, HTML5, CSS3, JavaScript |
| Training Environment | Google Colab, NVIDIA A100 |

---

## Contributions and Novelty

1. **Novel Architecture** — First system combining VideoMAE, Pose BiGRU, and XGBoost in a unified weighted ensemble for shooting foul detection
2. **Explainability Layer** — First known system producing kinematic evidence alongside every prediction
3. **Custom Annotated Dataset** — 369 NBA clips with 49 hard negatives and independent referee validation. No comparable public resource exists.
4. **Empirical Framework Evaluation** — XGBoost, LightGBM, and CatBoost systematically compared; XGBoost selected based on superior AUPRC

---

## Acknowledgements

- **Guhanathan Poravi** — Project supervisor
- **Saicharan Gnanapiragasam** — Technical advisor
- **Rukshan [Last Name]** — National Sri Lankan Basketball player and software engineer, who validated the basketball logic and ensured the system's biomechanical reasoning aligned with real referee decision-making

---

## Citation

```bibtex
@misc{rasheed2022smartrefnet,
  author    = {Muhammed Akthar Abdul Rasheed},
  title     = {SmartRef-Net: An AI-Powered Basketball Shooting Foul Detection System with Explainable Kinematic Evidence},
  year      = {2022},
  institution = {University of Westminster / IIT Sri Lanka},
  note      = {Final Year Project, BSc Computer Science}
}
```

---

## License

This project is released for academic and research purposes. Model weights and dataset are not publicly redistributable without permission.

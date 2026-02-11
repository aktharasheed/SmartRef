SmartRef: AI-Based Foul Detection Assistant for Basketball Referees
1. Project Overview

SmartRef is an AI-powered foul detection assistant designed to support basketball referees by automatically analyzing short video clips (5–15 seconds) and predicting whether a defensive contact foul has occurred.

The current Minimum Viable Product (MVP) focuses strictly on:

Binary Classification: Foul vs No-Foul (Defensive Contact Fouls Only)

Granular foul classification (charging vs blocking, role classification, non-contact violations) is intentionally deferred to future phases to ensure the binary system is reliable, stable, and accurate first.

2. Problem Statement

Basketball officiating is highly demanding:

Referees must make split-second decisions.

Player interactions occur in high-speed, occluded environments.

Contact is subtle and context-dependent.

Cognitive load is extremely high.

As a result:

Fouls are missed.

Calls are inconsistent.

Defensive contact fouls (blocking, holding, pushing) are especially controversial.

There is currently no robust end-to-end AI system that processes full short basketball video clips to classify foul/no-foul events using real game footage.

SmartRef addresses this gap.

3. Project Goal

To develop a working AI system that:

Processes short basketball clips (5–15 seconds)

Tracks players consistently across frames

Extracts pose and motion information

Predicts a binary outcome:

Foul

No Foul

Provides visual evidence to justify its decision

4. Dataset Strategy
Current Dataset (MVP Phase)

For the current MVP, the annotated dataset consists of:

100 short clips

Each clip: 5–15 seconds

All clips in 16:9 landscape format

The dataset is balanced across three categories:

Shooting fouls

Blocking fouls

Clean plays (no foul)

This balanced distribution ensures that the binary classifier does not become biased toward either the foul or no-foul class during training.

The clips are sourced from:

nbaplaydb.com (merged game footage)

YouTube NBA highlights

Each clip is manually annotated to include:

Offender bounding box (Red)

Defender bounding box (Blue)

Start frame (foul initiation)

Impact frame (primary contact moment)

Track-level attribute: foul_type = foul / no_foul

Future Dataset Expansion

After validating the MVP model on 100 clips, the dataset will be expanded to approximately:

600–700 total clips

200+ shooting fouls

200+ blocking fouls

200+ clean plays

This larger dataset will improve model generalization and support transition to more advanced foul-type classification in later phases.

5. Annotation Pipeline
Tool

Local CVAT installation (Docker-based)

Labeling Strategy

For each clip:

Draw tight bounding boxes around:

Offender (Red)

Defender (Blue)

Mark:

Start frame (when foul begins)

Impact frame (main contact moment)

Add track-level attribute:

foul_type = foul or no_foul

Export Format

COCO JSON

Includes:

Bounding boxes

Track IDs

Frame indices

Attributes

6. System Architecture

SmartRef follows a structured computer vision + deep learning pipeline.

Phase 1: Feature Extraction

This stage converts raw video into structured spatio-temporal features.

6.1 YOLO26 (Ultralytics – 2026 Release)

Pre-trained on COCO.

Function:

Detect players in each frame

Output bounding boxes

Output 17 pose keypoints per player

Benefits:

No initial training required

Real-time inference (30–100 FPS depending on variant)

Strong person detection and pose estimation out-of-the-box

6.2 ByteTrack (Multi-Object Tracking)

Integrated into Ultralytics pipeline.

Function:

Assign persistent player IDs across frames

Maintain track continuity during occlusion

Output player trajectories

Key Advantage:

No training required

Hyperparameter-based association tracker

Ideal for fast player interactions

6.3 Derived Features Per Clip

From YOLO26 + ByteTrack:

Keypoint sequences (pose over time)

Player trajectories

Relative distance between players

Velocity and acceleration

Contact proximity frames

Temporal interaction window

These features are used implicitly by the video classifier.

Phase 2: Binary Classification (Foul vs No-Foul)

Two candidate architectures are considered.

Option 1: SlowFast (Short-Term Prototype)

Developed by Meta AI.

Architecture:

Dual-pathway 3D CNN

Slow pathway → semantic understanding

Fast pathway → motion sensitivity

Pre-trained on:

Kinetics-400 / Kinetics-700

Fine-Tuning:

Replace final layer with binary sigmoid output

Train on 600 annotated clips

Strengths:

Fast training

Lower VRAM requirement

Strong short-term motion modeling

Expected Performance:

80–88% F1 score

15–30 FPS inference

Recommended for MVP.

Option 2: VideoMAE (Long-Term Accuracy Focus)

Masked Autoencoder Transformer (ViT-based).

Pre-training:

Self-supervised

Learns via masked video reconstruction

Data-efficient for small datasets

Fine-Tuning:

Replace classification head with binary sigmoid

Strengths:

Better generalization on small datasets

Strong attention maps (explainability)

Potentially higher accuracy

Expected Performance:

82–92% F1 score

10–20 FPS inference

Recommended for post-MVP improvement.

7. Final Output System

Input:

5–15 second basketball clip

Pipeline:
YOLO26 → ByteTrack → Video Model → Binary Output

Output:

Annotated video

Overlay:

"Foul: 87%" (Red)

"No Foul: 92%" (Green)

Bounding boxes on offender/defender

Pose skeletons

Highlighted start & impact frames

Deployment:

Google Colab notebook demo

Upload clip → Process → Download result

8. Why YOLO26 + ByteTrack?
YOLO26

Latest Ultralytics model (2026)

Pre-trained on COCO

Strong general person detection

No full retraining required

ByteTrack

No training required

Maintains identity consistency

Ideal for short contact events

Together:

Fast prototyping

Reduced training time

High performance without building models from scratch

9. Current Project Status (February 2026)

Proposal completed

CVAT local environment running

58+ annotated events completed

YOLO26 + ByteTrack inference operational

Binary classifier selection:

SlowFast → short-term

VideoMAE → long-term

10. Key Technical Strengths

End-to-end video-based analysis (not frame-only)

Uses temporal modeling (not static image classification)

Maintains player identity tracking

Produces explainable visual evidence

Modular and scalable architecture

11. Future Extensions (Post-MVP)

After binary model stability:

Offensive vs defensive role classification

Charging vs blocking classification

Contact vs non-contact separation

Rule-based integration for traveling/double dribble

Real-time referee dashboard integration

12. Expected Impact

SmartRef aims to:

Reduce referee cognitive load

Improve officiating consistency

Provide explainable AI support

Reduce controversy in competitive basketball

Lay foundation for AI-assisted refereeing systems
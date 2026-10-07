# Surgical Phase Recognition with ST-GCN on Cholec80

End-to-end computer vision system for recognizing surgical phases from laparoscopic videos using skeletal/keypoint sequences and a Spatial-Temporal Graph Convolutional Network (ST-GCN).

---

## Project Overview

This project implements a complete pipeline for **Surgical Phase Recognition**, a critical task in computer-assisted surgery, medical robotics, and operating room analytics.

Instead of relying only on appearance-based features, the system extracts keypoints from surgical frames and models their temporal evolution using an ST-GCN architecture.

### Key Features

- Keypoint extraction from surgical frames (MediaPipe + ORB fallback)
- Sliding-window sequence generation
- Skeleton/keypoint normalization
- Lightweight Spatial-Temporal Graph Convolutional Network (ST-GCN)
- Phase timeline visualization with confidence scores
- Gradio-based interactive demo

---

## Dataset

**Cholec80** (subset)

- Source: Publicly available Cholec80 frames + phase annotations
- Videos processed: 5
- Total frames with keypoints: 14,266
- Final windows created: 880
- Classes used in this version:  
  - `Preparation`  
  - `CalotTriangleDissection`

> Note: Only two dominant phases were present in the processed subset. The pipeline supports expansion to the full 7 Cholec80 phases.

---

## Pipeline Architecture

1. **Frame Loading & Exploration**
2. **Keypoint Extraction** (MediaPipe with ORB fallback for surgical scenes)
3. **Sequence Windowing** (48-frame windows, stride 16)
4. **Normalization** of keypoint sequences
5. **ST-GCN Model** for temporal phase classification
6. **Inference & Timeline Visualization**
7. **Gradio Deployment**

---

## Model Details

- **Architecture**: Custom Lightweight ST-GCN
- **Input Shape**: `(batch, 3, 48, 33)`
- **Parameters**: ~408K
- **Output**: Surgical phase class

### Training Results (5 Epochs)

| Metric                    | Value    |
|--------------------------|----------|
| Best Validation Accuracy | 82.95%   |
| Macro F1-Score           | 0.83     |

---

## Installation

```bash
pip install torch torchvision mediapipe opencv-python gradio scikit-learn matplotlib seaborn tqdm pillow
```

├── pose_data/                  # Extracted keypoint sequences

├── X_train.npy / X_val.npy

├── y_train.npy / y_val.npy

├── phase_mapping.json

├── best_stgcn_model.pth

├── Phase notebooks / scripts

└── README.md



##  Results & Observations

The complete pipeline works end-to-end.
Current model shows strong performance on the two dominant phases.
Severe class imbalance in the subset caused majority-class bias during inference.
Processing more videos and training longer will significantly improve phase diversity and robustness.


## Future Improvements

Process the full Cholec80 dataset (all 7 phases)
Replace ORB fallback with better surgical tool pose / instrument tracking
Add class balancing techniques (weighted loss, oversampling)
Experiment with Transformer-based temporal models
Real-time inference on video streams
Ergonomic risk / deviation scoring module


## Technologies Used

Python
PyTorch
MediaPipe
OpenCV
Gradio
Scikit-learn
Matplotlib / Seaborn


## Author
Computer Vision Project – Surgical Phase Recognition

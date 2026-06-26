# Ottermap Technical Challenge – Semantic Segmentation

## Overview

This project performs semantic segmentation on aerial imagery using **SegFormer-B0**.

The objective is to detect Turf/Grass regions from aerial images and export GIS-compatible outputs.

---
## Features

- Semantic segmentation using SegFormer-B0
- Binary turf/grass detection
- Automatic dataset loading
- Training and validation pipeline
- Model checkpoint saving
- Single-image inference
- GeoJSON polygon export
- Evaluation using Pixel Accuracy, IoU, and Dice Score
## Project Structure

```
ottermap-project/
│
├── config.py
├── dataset.py
├── train.py
├── inference.py
├── polygonize.py
├── metrics.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── images/
│   ├── labels/
│   ├── masks/
│   ├── train/
│   └── val/
│
├── outputs/
├── results/
├── weights/
└── scripts/
```

---
## Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- GeoPandas
- Rasterio
- NumPy
- Pillow
- tqdm

## Installation

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python train.py
```

The best model will be saved to:

```
weights/best_model.pth
```

---

## Inference

Predict on an image.

```bash
python inference.py --image data/images/1.jpg
```

Outputs are saved inside:

```
outputs/
```

---

## GeoJSON Export

Convert predicted masks to GeoJSON.

```bash
python polygonize.py
```

Results are saved inside:

```
results/
```

---

## Model

- SegFormer-B0
- Transfer Learning
- Binary Semantic Segmentation
- AdamW Optimizer

---

## Evaluation Metrics

- Pixel Accuracy
- IoU
- Dice Score

---

## Dataset Notes

During development, it was observed that the provided aerial imagery does not include georeferencing metadata while the annotations are stored as geographic coordinates. The preprocessing pipeline and debugging scripts were created to investigate and document this issue. The implementation is designed so that once correctly aligned imagery or spatial metadata is available, the training and inference pipeline can be executed without structural changes.

---
## Future Improvements

- Train the model using correctly georeferenced imagery once available.
- Experiment with larger SegFormer variants (B1/B2).
- Add data augmentation to improve generalization.
- Perform hyperparameter tuning.
- Support batch inference on multiple images.
- Export additional GIS formats such as Shapefile.
## Dataset

The original challenge dataset is **not included** in this repository.

To reproduce this project, place the dataset inside the `data/` directory while maintaining the expected folder structure described in this repository.

The dataset was excluded to keep the repository lightweight and to respect the challenge dataset distribution.

## Author

Abhimanyu Upadhyay
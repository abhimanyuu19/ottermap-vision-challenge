# Ottermap Vision/ML Engineer Technical Challenge

**Author:** Abhimanyu Upadhyay

## 1. Project Overview

This project implements a semantic segmentation pipeline to identify turf/grass regions from aerial imagery using the SegFormer-B0 transformer architecture. The objective is to generate binary segmentation masks and convert the predicted regions into GIS-compatible GeoJSON polygons.

The project was designed as a complete machine learning pipeline consisting of dataset preparation, model training, inference, evaluation, and polygon generation.

---

# 2. Problem Statement

The goal of the challenge is to detect turf regions from aerial imagery and export the predictions as polygons suitable for geographic information systems (GIS).

The solution includes:

* Dataset preparation
* Binary mask generation
* SegFormer training
* Model inference
* Polygon extraction
* GeoJSON export

---

# 3. Project Architecture

The workflow follows this pipeline:

Image → Dataset Loader → SegFormer → Prediction Mask → Polygon Extraction → GeoJSON

The implementation is modular so that each stage can be independently modified or extended.

---

# 4. Dataset Preparation

The dataset preparation process included:

* Organizing images and labels
* Generating binary masks
* Splitting data into training and validation sets
* Preparing data loaders for PyTorch

Custom preprocessing scripts were developed to automate these steps.

---

# 5. Model Selection

SegFormer-B0 was selected because it provides a good balance between segmentation accuracy and computational efficiency.

Advantages include:

* Transformer-based encoder
* Lightweight architecture
* Strong semantic segmentation performance
* Transfer learning support through Hugging Face Transformers

---

# 6. Training Pipeline

The training pipeline performs the following operations:

* Loads the dataset
* Preprocesses images
* Fine-tunes the pretrained SegFormer model
* Computes training loss
* Evaluates validation performance
* Saves the best model using validation IoU

AdamW optimizer is used during training.

---

# 7. Evaluation Metrics

The project evaluates model performance using:

* Pixel Accuracy
* Intersection over Union (IoU)
* Dice Score

These metrics provide complementary measures of segmentation quality.

---

# 8. Inference Pipeline

The inference pipeline:

* Loads a trained model
* Predicts a segmentation mask for a new image
* Saves the predicted binary mask
* Creates an overlay visualization
* Stores results in the outputs directory

---

# 9. Polygon Generation

Predicted masks are converted into polygons using OpenCV contour extraction.

The generated polygons are exported as GeoJSON files using GeoPandas, making them suitable for GIS applications.

---

# 10. Technical Investigation

During development, a spatial alignment issue was identified.

The provided GeoJSON annotations use geographic coordinates (EPSG:4326), while the supplied imagery did not contain georeferencing metadata required to directly align annotations with image pixels.

Several preprocessing and debugging steps were performed to investigate this behavior, including metadata inspection, visualization, and validation of coordinate reference systems.

A clarification request was sent to the challenge organizers before making unsupported assumptions about the data.

---

# 11. Future Improvements

Potential future enhancements include:

* Training with correctly georeferenced imagery
* Data augmentation
* Hyperparameter optimization
* Evaluation on additional datasets
* Batch inference support
* Export to additional GIS formats

---

# 12. Conclusion

This project demonstrates an end-to-end semantic segmentation workflow using modern deep learning techniques.

In addition to implementing the complete machine learning pipeline, emphasis was placed on reproducibility, modular code organization, and careful technical investigation of dataset-related issues.

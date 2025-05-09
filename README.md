# qbio460fp
Deep Learning Approaches for the Classification of Acute Lymphoblastic Leukemia in Peripheral Blood Smear Images

Deep Learning for Acute Lymphoblastic Leukemia (ALL) Subtype Classification

Overview

This project explores deep learning approaches to classify Peripheral Blood Smear (PBS) images into four diagnostic categories of Acute Lymphoblastic Leukemia (ALL):

Benign
Early Precursor
Pre-B
Pro-B
We compare a baseline CNN with several advanced architectures including ResNet-50, VGG16, EfficientNetB0, and ViT-B/16. We also apply Grad-CAM to evaluate model interpretability and highlight regions of influence on model predictions.

Key Features

Multi-class Leukemia Classification
Transfer Learning with Pretrained Models
Model Comparison Across CNNs and Vision Transformers
Explainable AI with Grad-CAM Visualization
Dataset

Leukemia Classification Dataset from Kaggle (Mehrad Aria, 2023)
224x224 RGB blood smear images labeled into four classes
Methods

Data Preprocessing & Augmentation
Model Training (CNN, ResNet-50, VGG16, EfficientNetB0, ViT-B/16)
Performance Evaluation (Accuracy, Precision, Recall, F1-score, ROC)
Model Explainability using Grad-CAM
Results

ViT-B/16 achieved the highest validation accuracy (99.49%).
Grad-CAM provided insights into model focus, revealing strengths and limitations in clinical interpretability.
Authors

Iliyan Hariyani
Iliyan Valani
Shoaib Valani
Advisors

Professor Tsu-Pei Chiu
Jesse Weller


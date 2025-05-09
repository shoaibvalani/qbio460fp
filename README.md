Deep Learning for Acute Lymphoblastic Leukemia (ALL) Subtype Classification

Abstract
--------
Acute Lymphoblastic Leukemia (ALL) is a common cancer requiring early and accurate diagnosis for effective treatment. 
In this project, we explored how deep learning can improve the diagnosis process by classifying blood smear images 
into four categories: Benign, Early Precursor, Pre-B, and Pro-B ALL. We compared a simple baseline CNN with several 
advanced models (ResNet-50, VGG16, EfficientNetB0, and ViT-B/16), applying transfer learning and fine-tuning strategies. 
All advanced architectures significantly outperformed the baseline CNN, with ViT-B/16 achieving the highest accuracy. 
To understand the decision-making process of these models, we applied Grad-CAM to visualize which image regions influenced 
the predictions. Our results highlight the potential of deep learning models to achieve more effective and consistent 
leukemia diagnosis.

Overview
--------
This project applies deep learning to classify Peripheral Blood Smear (PBS) images into four ALL subtypes:
- Benign
- Early Precursor
- Pre-B
- Pro-B

We compare a baseline CNN with advanced models:
- ResNet-50
- VGG16
- EfficientNetB0
- Vision Transformer (ViT-B/16)

We also use Grad-CAM to visualize model decision-making for interpretability.

Dataset
-------
Leukemia Classification Dataset (Kaggle, Mehrad Aria, 2023)
- 224x224 RGB blood smear images labeled into four classes.

Key Features
------------
- Multi-class image classification using CNNs and Transformers.
- Transfer learning with pretrained models.
- Grad-CAM visualizations for model interpretability.
- Comparative evaluation of model performance.

File Structure
--------------
- `EDA.ipynb` – Exploratory data analysis.
- `Augmentation.ipynb` – Data augmentation to address class imbalance.
- `DimensionalityReduction.ipynb` – PCA and t-SNE visualizations.
- `CNN.ipynb` – Baseline CNN model (built from scratch).
- `basic_cnn_seq_model.h5` – Saved weights for the baseline CNN.
- `ResNet50.ipynb` – ResNet-50 fine-tuning and evaluation.
- `resnet50_augmented_model.h5` – Saved ResNet-50 model weights.
- `VGG.ipynb` – VGG16 model training and evaluation.
- `vgg_augmented_model.h5` – Saved VGG16 model weights.
- `EfficientNet.ipynb` – EfficientNetB0 model training and fine-tuning.
- `efficientnetb0_model.h5` – Saved EfficientNetB0 model weights.
- `VisionTransformer.ipynb` – ViT-B/16 model training using PyTorch.
- `vit_model.pth` – Saved ViT-B/16 model weights.
- `GradCAM.ipynb` – Grad-CAM implementation and visualizations.
- `Original/` – Original dataset folder (images organized by class).
- `Segmented/` – Optional segmented cell images (not used in study).

Team
----
- Iliyan Hariyani
- Iliyan Valani
- Shoaib Valani

Advisors
--------
- Prof. Tsu-Pei Chiu
- Jesse Weller

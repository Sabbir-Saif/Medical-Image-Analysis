# Medical Image Classification for Brain Tumor Detection
This project focuses on deep learning-based medical image classification, specifically for detecting and classifying brain tumors using MRI scans. The goal is to develop an automated, efficient, and accurate system that can assist radiologists and medical professionals in diagnosing different types of brain tumors.

The project leverages transfer learning with three well-known deep learning architectures—InceptionV3, VGG16, and ResNet50.
to classify MRI images into four categories:

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

# Methodology

i.   Extract images from a compressed dataset.
ii.  Resize all images to a consistent shape (128×128 pixels).
iii. Convert images to RGB format and store them as NumPy arrays.
iv.  Split the dataset into training, validation, and test sets.
v.   Normalize image pixel values using the appropriate preprocessing function.

vi.  Implement three pre-trained models (InceptionV3, VGG16, ResNet50) using transfer learning. Freeze initial layers and add custom fully connected layers for classification.
vii. Train each model on the dataset with categorical cross-entropy loss and Adam optimizer.Evaluate models on a test dataset and evaluate performances
viii.Generate classification reports with precision, recall, and F1-score and plot confusion matrices to visualize misclassifications.

# Significance of the Project

Medical Impact: Helps automate brain tumor detection, reducing diagnostic time and improving accuracy.
Deep Learning Application: Demonstrates the power of CNN-based transfer learning in medical imaging.
Comparative Analysis: Evaluates the performance of different state-of-the-art architectures in MRI classification.

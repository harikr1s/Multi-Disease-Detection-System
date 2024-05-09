# Multi-Disease Detection System

This Streamlit application is designed for multi-disease detection using machine learning models trained with Teachable Machine. It allows users to upload medical images for prediction and provides insights such as disease classification and saliency maps.

## Overview

The application is built using Python and Streamlit, providing an interactive and user-friendly interface for medical image analysis. It incorporates deep learning models trained on diverse medical datasets to predict diseases such as brain tumors, lung pneumonia, and kidney cancer from corresponding MRI, X-ray, and CT scan images.

## Models and Datasets

The machine learning models used in this application were trained using Google's Teachable Machine platform. The datasets used for training these models are publicly available on Kaggle:

- [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri): Contains MRI images of brain tumors classified into different categories.
- [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset): Includes various medical scan images for classification tasks.
- [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia): Consists of X-ray images for detecting pneumonia in the lungs.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/multi-disease-detection.git
   cd multi-disease-detection
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
3. Run the streamlit application:
   ```bash
   streamlit run main.py

## Usage

1. Select the disease type (Brain, Lung, or Kidney) from the sidebar.
2. Upload a corresponding medical image (MRI, X-ray, or CT scan) for prediction.
3. View the prediction results, including the disease classification and confidence score.
4. Explore the saliency map to understand the model's attention areas in the input image.

## Features

- Responsive UI with sidebar navigation for disease selection.
- Real-time image upload and prediction using trained machine learning models.
- Display of prediction results, confidence scores, and saliency maps for interpretability.
- Background customization and visual enhancements for a better user experience.

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- Streamlit
- Pillow
- NumPy
- Matplotlib

## Acknowledgements

- Google's Teachable Machine for model training capabilities.
- Kaggle for providing diverse and labeled medical datasets for training and testing.
- Streamlit for enabling rapid development and deployment of interactive web applications.
- The background images used in this project were sourced from [IlNaz Ismagilov on Vecteezy](https://www.vecteezy.com/members/ilnaz-ismagilov-201529069)






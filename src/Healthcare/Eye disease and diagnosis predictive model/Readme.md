  # Eye Disease Prediction and Diagnosis using CNN
  # Project Overview

This project presents a deep learning-based system for the automatic detection and diagnosis of eye diseases using Convolutional Neural Networks (CNNs). The aim is to assist in early diagnosis by analyzing retinal and eye images, which can significantly reduce the chances of vision loss if diseases are detected on time.

**The solution is designed to**:

Provide fast and accurate predictions of eye conditions.

Support medical professionals by offering an AI-assisted pre-diagnosis tool.

Enhance accessibility for people in remote or resource-limited areas where regular eye checkups may not be available.

Reduce manual screening efforts, enabling faster medical decision-making.

The model is trained on a curated dataset of eye disease images. By applying data cleaning, rescaling, augmentation, and visualization, the dataset was prepared for robust training. The trained CNN can classify eye images into different categories, predict whether the eye is healthy or diseased, and suggest a preliminary diagnosis.

This project highlights how AI in healthcare can bridge the gap between patients and medical professionals by providing an initial screening system that helps in early detection and treatment planning.

# Key Features

**Data Preprocessing**: Cleaned and rescaled image data for consistency.

**Data Augmentation**: Applied rotation, zoom, and flip transformations to improve model generalization.

**Data Visualization**: Displayed sample images to understand dataset distribution.

**Model Training**: Developed a CNN model using TensorFlow/Keras for multi-class classification.

**Prediction on New Data**: Tested the model’s performance on unseen eye images.

**Performance Evaluation**: Visualized results using a confusion matrix.

# Technologies Used

Python

TensorFlow / Keras

NumPy & Pandas

Matplotlib & Seaborn (for visualization)

## Workflow

Load and clean the dataset.

Rescale images for uniformity.

Apply augmentation to increase dataset variability.

Visualize dataset samples.

Train CNN model on the processed dataset.

Predict eye disease from test images.

Evaluate with confusion matrix and accuracy metrics.

## Model Performance

Validation Loss: 0.6137

Validation Accuracy: 77.16%

## Sample Prediction & Diagnosis

**Prediction**: Normal

**Diagnosis**: The eye appears healthy and no signs of disease are detected. Continue routine eye checkups.

**Note**: Please consult a doctor for a more accurate and professional diagnosis.

## Sample Visualizations

Class distribution plots

Augmented images

Confusion matrix for performance

##  Future Scope

Deploy the model as a web application (e.g., using Streamlit).

Extend to real-time detection via webcam integration.

Improve accuracy with transfer learning models like VGG16, ResNet, or EfficientNet.

## Author

**Smriti Pandey**
 **B.Tech in Computer Science & Engineering (AIML Specialization)**
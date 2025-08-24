 #  Fruit Disease Detection & Diagnosis Model

## Project Overview
This project is a **deep learning-based fruit disease classification system** that predicts both:
1. **The fruit type** (e.g., Mango, Guava, Apple, Pomegranate)  
2. **The disease affecting it** (or Healthy status)  

It uses **Convolutional Neural Networks (CNN)** with **MobileNetV2** as the base model for efficient feature extraction.  
The model not only predicts the class but also provides **diagnostic information** about the detected disease.

---

##  Dataset
The dataset used for training contains images of **Apple, Guava, Mango, and Pomegranate** each categorized into **Healthy** and various disease classes.  
The dataset structure follows the format:
Dataset/
- Blotch_Apple/
- Anthracnose_Guava/
- Alternaria_Mango/
- Healthy_Mango/
 ...


**Dataset Source:**  
[Fruits Dataset for Fruit Disease Classification - Kaggle](https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification?utm)

---

##  Data Analysis
1. **Class Count Visualization**  Bar plots were created to check **dataset imbalance** across all classes.  
2. **Data Augmentation**  Applied transformations such as:
   - Rotation
   - Zoom
   - Horizontal Flip
   - Width & Height Shift  
   This increased dataset diversity and helped prevent overfitting.

---

##  Libraries Used
- **TensorFlow**  For building and training the CNN model  
- **Keras**  High-level API for TensorFlow model creation  
- **NumPy** Numerical computations  
- **Matplotlib** Data visualization (class count plot, training curves)  
- **Pillow (PIL)**  Image preprocessing  
- **Scikit-learn**  Train-test split, evaluation metrics  
- **OS & Shutil**  File handling and dataset management

---

##  Model Architecture
The model is based on **MobileNetV2** (pre-trained on ImageNet):
- **Base Model**: MobileNetV2 (frozen initial layers for transfer learning)  
- **Global Average Pooling**: Reduces feature map size while retaining information  
- **Dense Layer (128 units)**: Fully connected layer with ReLU activation  
- **Dropout (0.3)**: Prevents overfitting  
- **Output Layer**: Softmax activation with `n_classes` neurons  

---

##  Training Process
1. **Data Preprocessing**:
   - Resized images to **224×224 pixels**
   - Normalized pixel values to `[0,1]`
   - Applied **data augmentation**
   
2. **Transfer Learning**:
   - Used MobileNetV2 with frozen convolutional layers
   - Added custom layers for classification

3. **Fine-Tuning**:
   - Unfrozen some top layers of MobileNetV2
   - Retrained with a low learning rate for better accuracy

4. **Evaluation**:
   - Achieved **~86% accuracy** on the test dataset
   - Used metrics: Accuracy, Precision, Recall, F1-Score

---

##  Prediction & Diagnosis
- The model predicts **Fruit Type** and **Disease Name** from the image
- A **diagnosis dictionary** maps diseases to their causes and effects  
Example Output:
Fruit: Mango
Disease: Stem and Rot (Lasiodiplodia)
Diagnosis: Caused by Lasiodiplodia, it affects stems and leads to rapid rotting.

##  Usage
1. **Train the Model** Load dataset, preprocess, train using MobileNetV2.
2. **Save the Model**  Export as `.h5` or `.keras` file.
3. **Load & Predict**  Pass a fruit image to get predictions + diagnosis.

## **Author**
**Smriti Pandey**  
BTech CSE (Specialization in AIML)  
Passionate about AI, Machine Learning  projects.  


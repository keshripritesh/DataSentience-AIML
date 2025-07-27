🧠 Toxic Comment Detector
This project implements a robust, multi-label NLP classifier designed to identify and flag various forms of toxic behavior in user-generated online comments. By automatically detecting harmful language, it aims to enhance online community safety and foster healthier digital environments.

📌 Overview
Utilizing data from the Jigsaw Toxic Comment Classification Challenge on Kaggle, this solution employs a foundational text classification approach to categorize comments across multiple toxicity dimensions simultaneously.

[!ui screenshot](assets/Screenshot_7-7-2025_22953_localhost.jpeg)
[!ui screenshot](assets/Screenshot_7-7-2025_23024_localhost.jpeg)


💡 Key Features
Multi-Label Classification: Accurately identifies comments exhibiting one or more toxic attributes, including toxic, severe_toxic, obscene, threat, insult, and identity_hate.

Scalable Baseline Model: Leverages TF-IDF vectorization paired with Logistic Regression, providing an effective and interpretable starting point for text classification.

Modular Architecture: Designed with clear separation of concerns for data preprocessing, model training, and prediction, facilitating maintainability and future enhancements.

Extensible Design: Built to easily integrate more advanced neural network architectures, such as Transformer models (e.g., BERT, RoBERTa), for potential performance improvements.

User-Friendly Interface: Includes a clean Command-Line Interface (CLI) for straightforward demonstration and testing of the model's capabilities.
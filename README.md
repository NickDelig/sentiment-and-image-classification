# 🧠 Machine Learning & Deep Learning Portfolio

Welcome to my AI portfolio project! This repository contains a comprehensive Jupyter Notebook demonstrating my ability to build, train, and evaluate both traditional machine learning models and modern deep neural networks using Python, Scikit-Learn, and PyTorch. 

The project is split into three main sections covering Natural Language Processing (NLP) and Computer Vision (CV).

## 🛠️ Technologies & Frameworks Used
* **Languages:** Python
* **Deep Learning:** PyTorch, Torchvision
* **Machine Learning:** Scikit-Learn
* **Data Processing & Utilities:** TensorFlow/Keras (for dataset loading), NumPy, Matplotlib
* **Techniques:** Transfer Learning, Data Augmentation, Feature Selection (Information Gain), Word Embeddings (GloVe)

---

## 📊 Project Breakdown

### Part A: Traditional ML for Sentiment Analysis (IMDB Dataset)
**Goal:** Classify movie reviews as positive or negative using traditional ML techniques.
* **Feature Engineering:** Built a custom vocabulary using **Information Gain** (`mutual_info_classif`) to select the top 1,000 most informative words.
* **Models:** Implemented and compared **Logistic Regression** (with L2 Regularization) and **Bernoulli Naive Bayes**.
* **Results:** Plotted learning curves to visualize F1-score progression and achieved ~86% accuracy on the test set.

### Part B: Deep Learning for NLP (Stacked Bi-LSTM)
**Goal:** Improve sentiment analysis using a Recurrent Neural Network.
* **Architecture:** Designed a **Stacked Bidirectional LSTM** (2 layers, 128 hidden units) using PyTorch.
* **Embeddings:** Integrated pre-trained **GloVe (100d)** word embeddings to map the vocabulary.
* **Optimization:** Leveraged GPU acceleration, Global Max Pooling, and Dropout for regularization. 
* **Results:** Achieved ~85% accuracy with highly stable validation loss curves.

### Part C: Computer Vision & Transfer Learning (FashionMNIST)
**Goal:** Classify images of clothing into 10 categories using Convolutional Neural Networks (CNNs).
* **Data Pipeline:** Implemented data augmentation (Random Horizontal Flips, Random Rotations) to improve model robustness.
* **Transfer Learning:** Fine-tuned a pre-trained **ResNet18** model.
    * *Custom Modification 1:* Adjusted the first convolutional layer to accept 1-channel grayscale images instead of the standard 3-channel RGB.
    * *Custom Modification 2:* Replaced the final fully connected layer with a custom Multi-Layer Perceptron (MLP) head with Dropout.
* **Results:** Achieved over **90% accuracy** on the test set, with excellent Precision/Recall scores across all 10 clothing categories.

---

## 🚀 How to Run the Code
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed: `pip install torch torchvision scikit-learn numpy matplotlib tensorflow`
3. Open `Ai_Final.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells sequentially. *Note: Part B will automatically download the required GloVe embeddings if they are not found in the directory.*

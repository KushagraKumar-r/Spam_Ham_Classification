# 📧 Spam vs. Ham Email Classification

This repository contains a machine learning project that classifies email messages as **spam** or **ham** (non-spam). It leverages natural language processing (NLP) techniques to clean, transform, and model textual email data for binary classification.

## 🚀 Project Overview

Spam detection is a common NLP problem with real-world applications in email filtering, SMS screening, and messaging platforms. This project includes:

- Text preprocessing (tokenization, stopword removal, stemming/lemmatization)
- Feature extraction using TF-IDF or CountVectorizer
- Model training using classifiers like Naive Bayes, Logistic Regression, or SVM
- Evaluation using accuracy, precision, recall, and confusion matrix

## 🧠 Algorithms Used

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machines (optional)
- Random Forest (optional)

## 📁 Project Structure

spam-ham-classification/
├── data/ # Raw and processed datasets
│ └── spam.csv
├── notebooks/ # Jupyter notebooks for EDA and model training
├── src/ # Source code for preprocessing, modeling, and evaluation
├── models/ # Saved model files
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── main.py # Main script for running the model


## 📊 Dataset

The dataset used is the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository, consisting of 5,000+ labeled SMS messages.

- [UCI SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

Each message is labeled as:
- `spam` – unsolicited or unwanted messages
- `ham` – legitimate messages

## ⚙️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-ham-classification.git
   cd spam-ham-classification
🧪 Evaluation Metrics
The model performance is evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

📌 Requirements
Python 3.7+

scikit-learn

pandas

numpy

matplotlib / seaborn

nltk / spaCy

📷 Sample Output

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🙌 Acknowledgments
UCI Machine Learning Repository

Scikit-learn documentation

NLTK for text preprocessing


# Textual Deepfake Detection: AI-Generated Product Review Detection

## Overview

This project uses Natural Language Processing (NLP) and machine learning to detect whether product reviews are human-written or AI-generated.

The goal is to support online review moderation by automatically flagging suspicious AI-generated reviews for further human review.

## Business Problem

Online product reviews influence customer trust, seller reputation, and purchasing decisions. With the rise of generative AI, fake reviews can be created more easily and may mislead customers or manipulate product ratings.

This project addresses the problem by building a supervised text classification pipeline that predicts whether a review is original/human-written or computer-generated.

## Dataset

Dataset: Fake Reviews Dataset
Source: Kaggle
Target classes:

* 0 = Human-written / Original Review
* 1 = AI-generated / Computer-Generated Review

## Methods

The project includes:

* Data loading and inspection
* Text preprocessing
* Label encoding
* TF-IDF feature extraction
* Traditional machine learning models
* Transformer-based DistilBERT model
* Model evaluation using accuracy, precision, recall, F1-score, classification reports, and confusion matrices

## Models Compared

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes
* DistilBERT

## Results

DistilBERT achieved the highest overall performance, outperforming the traditional machine learning models.

Linear SVM was the strongest traditional model and remains a practical option for faster and simpler deployment.

## Tools & Technologies

* Python
* pandas
* NumPy
* scikit-learn
* TF-IDF
* Logistic Regression
* Linear SVM
* Naive Bayes
* DistilBERT
* Hugging Face Transformers
* Matplotlib
* Google Colab / Jupyter Notebook

## Project Structure

```text
textual-deepfake-detection-nlp/
├── README.md
├── notebooks/
│   └── textual_deepfake_detection.ipynb
├── reports/
│   └── textual_deepfake_detection.html
├── images/
│   └── confusion_matrix.png
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone the repository.
2. Install the required Python packages.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Run the notebook cells in order.

## Future Improvements

* Test the model on reviews from other domains
* Add more transformer models for comparison
* Include reviewer metadata and behavioural features
* Build a simple web app for review classification
* Add regular retraining to adapt to new AI-generated writing styles

## Author

Zainab Alzubair
Data Science, AI, and Digital Marketing student based in Berlin
LinkedIn: https://www.linkedin.com/in/zainab-alzubair/

# Spam Classifier with Machine Learning

This project is a machine learning spam classifier built with Python.

The model analyzes text messages and classifies them as either spam or ham (normal message).

---

## Features

- SMS spam classification  
- Text processing with TF-IDF vectorization  
- Machine Learning model using Multinomial Naive Bayes  
- Accuracy evaluation  
- Classification report (precision, recall, f1-score, support)  
- Manual message testing  
- Prediction probability output  

---

## Technologies

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF Vectorizer  
- Multinomial Naive Bayes  

---

## How it works

The project follows a complete machine learning pipeline:

1. Load the dataset  
2. Select and rename relevant columns  
3. Split data into training and testing sets  
4. Convert text into numerical features using TF-IDF  
5. Train a classification model  
6. Evaluate performance  
7. Predict new messages  

---

## Dataset

This project uses the SMS Spam Collection dataset.

The dataset contains:

- `label`: spam or ham  
- `message`: SMS text  

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the project:

```
python main.py
```

Then enter a message to classify.

### Example

```
Enter a message: Congratulations! You won a free prize
Prediction: spam

ham: 2.40%
spam: 97.60%
```

---

## Learning Goals

This project was built to practice:

- Supervised Machine Learning  
- Text classification  
- Natural Language Processing (NLP) basics  
- Model evaluation  
- Building end-to-end ML pipelines  

---

## Author

Alef Santos

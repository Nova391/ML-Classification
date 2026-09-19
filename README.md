# ML Classification

A small machine learning text-classification project built with Python and scikit-learn.

The goal of this project was to learn the workflow of a practical machine learning classification system, from preparing text data to training a model and evaluating its predictions.

## What This Project Does

The model classifies customer-style messages into two categories:

* `issue`
* `inquiry`

Examples of issues include broken products, application crashes, login problems, and failed payments.

Examples of inquiries include store hours, return policies, pricing, shipping, and support information.

## Machine Learning Pipeline

The project follows this workflow:

**Dataset → Train/Test Split → TF-IDF → Logistic Regression → Predictions → Evaluation**

### 1. Dataset

The dataset is created in Python using a Pandas DataFrame containing example messages and their corresponding labels.

### 2. Train/Test Split

The dataset is divided into training and testing data using `train_test_split`.

### 3. TF-IDF

`TfidfVectorizer` converts the text messages into numerical feature vectors that the machine learning model can use.

### 4. Logistic Regression

A Logistic Regression classifier is trained on the TF-IDF features.

### 5. Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report

### 6. Custom Predictions

After training, the model is also tested on new messages that were not part of the original dataset.

For example:

* `"my app keeps crashing when I open it"` → `issue`
* `"can you tell me your store hours for tomorrow"` → `inquiry`

## Example Result

Using the current small dataset and test split, the model achieved:

**Accuracy: 83.3%**

The model correctly classified both custom test messages.

Because the dataset contains only a small number of examples, this accuracy should be considered a learning result rather than a meaningful measurement of real-world performance.

## Technologies

* Python
* Pandas
* scikit-learn
* TF-IDF
* Logistic Regression
* Git
* GitHub

## What I Learned

This project helped me understand a practical machine learning workflow and how text can be transformed into numerical features before being used by a classifier.

I also worked with train/test data, feature extraction, classification, prediction, and model evaluation.

This project is part of my broader goal of learning AI from the fundamentals and eventually building larger AI systems.

## Project Status

Completed as a practical machine learning learning project.

The next stage of my development is focused on Python automation, APIs, and AI-powered automation projects for my portfolio.

## Author

Mohamed

## Note

This README was written with the assistance of **ChatGPT**, which I use as an AI learning mentor throughout this project. The code and implementation were written by me as part of my learning process.

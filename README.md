# Fake News Detection System

This Python project implements a system to classify news articles as either "Fake News" or "Real News" using several machine learning classification models. The system processes text data from two CSV files, trains and evaluates these models, and provides a functionality for manually testing new, unseen news articles.

## Overview

The core steps involved in this project are:

1.  **Data Handling:** Loading and preprocessing datasets of fake and real news articles using the `pandas` library.
2.  **Text Preprocessing:** Cleaning and normalizing the text data using regular expressions and string manipulation.
3.  **Feature Extraction:** Converting the text data into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
4.  **Model Training:** Training four different machine learning classification models: Logistic Regression, Decision Tree Classifier, Gradient Boosting Classifier, and Random Forest Classifier.
5.  **Model Evaluation:** Assessing the performance of each trained model using accuracy scores and classification reports on a held-out test dataset.
6.  **Manual Testing:** Providing a function to input new news text and get predictions from all the trained models.

## Libraries Used

* **pandas (pd):** For efficient data manipulation and working with DataFrames.
* **numpy (np):** For performing numerical computations.
* **seaborn (sns):** For creating statistical visualizations (though not heavily used in this snippet).
* **matplotlib.pyplot (plt):** For generating plots and graphs (though not explicitly used in this snippet).
* **scikit-learn (sklearn):** A comprehensive machine learning library used for:
    * `model_selection.train_test_split`: Splitting the dataset into training and testing sets.
    * `metrics.accuracy_score`: Calculating the accuracy of the classification models.
    * `metrics.classification_report`: Providing detailed classification metrics (precision, recall, F1-score).
    * `feature_extraction.text.TfidfVectorizer`: Converting text data into TF-IDF feature vectors.
    * `linear_model.LogisticRegression`: Implementing the Logistic Regression classification model.
    * `tree.DecisionTreeClassifier`: Implementing the Decision Tree classification model.
    * `ensemble.GradientBoostingClassifier`: Implementing the Gradient Boosting classification model.
    * `ensemble.RandomForestClassifier`: Implementing the Random Forest classification model.
* **re:** For working with regular expressions to clean text data.
* **string:** For accessing string constants, particularly for punctuation removal.

## Data Loading and Preprocessing

1.  The script loads two CSV files: `fake.csv` (labeled as class 0) and `true.csv` (labeled as class 1).
2.  The first few rows of each dataset are displayed for inspection.
3.  A 'class' column is added to each DataFrame to indicate whether the news is fake (0) or real (1).
4.  The last 10 rows from each DataFrame are extracted for manual testing purposes and then removed from the main datasets.
5.  The fake and real news DataFrames are merged into a single DataFrame.
6.  Irrelevant columns ('title', 'subject', 'date') are dropped, focusing on the 'text' content for classification.
7.  Missing values are checked.
8.  The merged DataFrame is shuffled randomly to ensure unbiased training and testing splits.
9.  The index of the shuffled DataFrame is reset, and the old index column is removed.

## Text Preprocessing (`wordopt` Function)

The `wordopt` function performs the following text cleaning steps on each news article:

* Converts the text to lowercase.
* Removes text enclosed in square brackets (`[...]`).
* Replaces non-alphanumeric characters with spaces.
* Removes URLs (both `http` and `www`).
* Removes HTML-like tags (`<.*?>+`).
* Removes punctuation marks.
* Removes newline characters (`\n`).
* Removes words containing digits.

This function is applied to the 'text' column of the DataFrame.

## Data Splitting

The preprocessed text data (`x`) and their corresponding class labels (`y`) are split into training and testing sets using `train_test_split`, with 25% of the data reserved for testing.

## Text Vectorization (TF-IDF)

The `TfidfVectorizer` is used to convert the text data in both the training and testing sets into numerical TF-IDF vectors. This process weighs the importance of words based on their frequency in a document and across the entire corpus. The vectorizer is fitted only on the training data to prevent data leakage from the test set.

## Model Building and Evaluation

The script trains and evaluates four different classification models:

* **Logistic Regression:** A linear model used for binary classification.
* **Decision Tree Classifier:** A tree-based model that makes decisions based on feature values.
* **Gradient Boosting Classifier:** An ensemble method that builds multiple weak learners sequentially.
* **Random Forest Classifier:** An ensemble method that builds multiple decision trees and averages their predictions.

For each model:

1.  An instance of the model is created.
2.  The model is trained using the TF-IDF transformed training data (`xv_train` and `y_train`).
3.  Predictions are made on the TF-IDF transformed testing data (`xv_test`).
4.  The accuracy of the model is calculated and printed.
5.  A classification report, including precision, recall, F1-score, and support for each class, is printed.

## Output Label Function (`output_lable`)

This simple function takes a numerical class label (0 or 1) and returns its corresponding human-readable string ("Fake News" or "Real News").

## Manual Testing (`manual_testing` Function)

This function allows users to input a new piece of news text and see the predictions from all four trained models. It preprocesses the input text, transforms it into a TF-IDF vector using the **already fitted** `vectorization` object, and then uses each model to predict its class. The predictions are then printed in a user-friendly format.

## Usage

To test the system with a new news article, simply run the script. It will prompt you to enter the news text. After you enter the text, the predictions from the Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest models will be displayed.

The script also prints the final accuracy scores of all the trained models on the test dataset at the end of the execution.

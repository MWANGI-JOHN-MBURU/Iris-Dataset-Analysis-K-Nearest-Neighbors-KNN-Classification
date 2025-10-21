# Iris-Dataset-Analysis-K-Nearest-Neighbors-KNN-Classification

## 🌸 Iris Dataset Analysis & K-Nearest Neighbors (KNN) Classification

This project demonstrates data exploration and classification using the famous Iris flower dataset.
The goal is to understand the dataset structure, explore feature relationships, and implement a K-Nearest Neighbors (KNN) model for species classification.

## 📘 Project Overview

The Iris dataset is a well-known dataset in machine learning and statistics.
It contains measurements of iris flowers from three species — Setosa, Versicolor, and Virginica — and is often used for classification practice.

In this project:

The dataset is loaded using scikit-learn.

Data exploration and visualization are performed using Pandas, Matplotlib, and Seaborn.

A KNN classifier is implemented to classify flower species based on their features.

## 🧠 Steps Performed

Import Libraries
Essential libraries such as pandas, matplotlib, seaborn, and scikit-learn were imported for data handling, visualization, and modeling.

Load the Dataset

```
from sklearn.datasets import load_iris
df = load_iris(as_frame=True).frame
```


Explore the Data

Checked dataset dimensions using 
```
df.shape
```

Examined column details and data types with
```
df.info()
```

Generated statistical summaries with df.describe()

Visualize Feature Correlations
A correlation heatmap was created to show relationships between features:

```
import seaborn as sns
sns.heatmap(data=df.corr(), annot=True, cmap="Blues")
plt.title("Feature Correlation Heatmap")
plt.show()
```


Model Training (KNN)

``` 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```


Model Evaluation
The model’s performance can be evaluated using metrics such as accuracy, confusion matrix, and classification report.

## 📊 Visualization Example

The heatmap below illustrates how features like petal length and petal width have a strong correlation, making them powerful predictors in classification.
```
sns.heatmap(data=df.corr(), annot=True, cmap="Blues")
```

## 🧩 Tools & Libraries

Python

Pandas

Matplotlib

Seaborn

Scikit-learn

## 🚀 Future Enhancements

Compare KNN performance with other classifiers (e.g., SVM, Decision Tree, Logistic Regression).

Apply feature scaling and cross-validation to optimize accuracy.

Create interactive visualizations using Plotly.

## 💡 Author

John Mwangi
Data Science & Machine Learning Enthusiast

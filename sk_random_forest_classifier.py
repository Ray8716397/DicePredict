# coding=utf-8
# @Time : 2023/2/24 下午5:29
# @File : sk_random_forest_classifier.py
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
history = [[5, 5, 1], [6, 5, 4], [5, 5, 5], [4, 4, 1], [5, 4, 3], [6, 4, 2], [6, 3, 1], [5, 2, 1], [3, 2, 1], [5, 2, 1],
           [6, 3, 3], [6, 1, 1], [3, 2, 1], [6, 5, 4], [6, 4, 1], [5, 5, 4], [5, 2, 2], [5, 3, 2], [6, 4, 3], [6, 5, 3],
           [6, 6, 5], [4, 4, 2], [6, 4, 2], [5, 4, 3], [6, 4, 1], [6, 5, 3], [5, 2, 1], [6, 5, 3], [5, 5, 2], [6, 5, 1],
           [6, 3, 1], [6, 5, 4], [6, 1, 1], [3, 1, 1], [4, 3, 1], [3, 2, 1], [3, 2, 1], [4, 3, 2], [6, 6, 4], [4, 1, 1],
           [4, 3, 3], [3, 3, 3], [6, 4, 2], [6, 4, 2], [6, 2, 1], [6, 2, 1], [4, 4, 3], [6, 6, 3]
           ]

# 特征工程：将历史点数转换为特征
X = []
y = []
for i in range(len(history) - 2):
    X.append(history[i])
    y.append(history[i + 1])

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=sum(random.choice(history)))

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=sum(random.choice(history)))
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict the result of a game
result = clf.predict(np.array(history[-1]))
print("Predicted result:", result)

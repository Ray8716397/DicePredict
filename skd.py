# coding=utf-8
# @Time : 2023/2/23 下午2:28
# @File : skd.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load the training data
# data = np.loadtxt('sicbo_data.csv', delimiter=',')
history = [[3, 1, 1], [6, 5, 2], [4, 4, 4], [6, 3, 1], [6, 2, 1], [5, 3, 2], [5, 4, 3], [6, 4, 3], [6, 4, 2], [5, 3, 2],
           [5, 4, 1], [5, 3, 1], [3, 2, 1], [6, 5, 4], [4, 4, 2], [6, 5, 4], [3, 3, 1], [5, 2, 1], [6, 6, 3], [6, 4, 3],
           [4, 3, 3], [6, 4, 4], [2, 2, 1], [6, 5, 1], [4, 2, 2], [5, 2, 1], [2, 2, 1], [6, 3, 1], [4, 3, 1], [6, 2, 1],
           [4, 4, 1], [5, 1, 1], [6, 5, 3], [5, 2, 1], [6, 6, 4], [4, 3, 3], [3, 2, 2], [5, 5, 2], [6, 6, 2], [6, 3, 2],
           [3, 2, 2], [6, 5, 2], [4, 3, 2], [2, 1, 1], [5, 3, 2], [5, 3, 2], [3, 2, 2], [2, 2, 1], [5, 3, 2], [6, 5, 4],
           [6, 4, 4], [6, 3, 1], [4, 3, 2], [6, 4, 2], [6, 4, 4], [4, 4, 3], [5, 5, 5], [4, 3, 1], [2, 2, 2], [5, 4, 3],
           [6, 1, 1], [5, 2, 2], [2, 1, 1], [4, 2, 2], [6, 5, 1], [5, 4, 1], [6, 6, 1], [3, 5, 1], [6, 4, 2], [6, 4, 3],
           [6, 4, 2], [4, 3, 1], [6, 3, 1], [5, 4, 1]]
data = np.array(history)
# Split the data into input (X) and target (y) arrays
X = []
y = []
for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

# Create a decision tree classifier and fit the data
model = DecisionTreeClassifier()
model.fit(X, y)

# Get input from user (three dice rolls)
rolls = [5, 4, 1]

# Make a prediction
pred = model.predict([rolls])[0]
# Print the prediction
print("The predicted outcome is:", pred)

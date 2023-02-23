# coding=utf-8
# @Time : 2023/2/23 下午2:27
# @File : ptd2.py
import torch
import torch.nn as nn
import numpy as np

# Define the neural network
class SicBoModel(nn.Module):
    def __init__(self):
        super(SicBoModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = SicBoModel()
model.load_state_dict(torch.load('sicbo_model.pth'))

# Get input from user (three dice rolls)
rolls = input("Enter three dice rolls separated by commas: ")
rolls = [int(x) for x in rolls.split(",")]

# Convert input to tensor
x = torch.tensor([rolls], dtype=torch.float)

# Make a prediction
with torch.no_grad():
    output = model(x)
    pred = torch.argmax(output, dim=1)

# Print the prediction
print("The predicted outcome is:", pred.item()+4)

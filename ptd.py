import torch
import torch.nn as nn
import numpy as np


# Define the neural network
class DiceModel(nn.Module):
    def __init__(self):
        super(DiceModel, self).__init__()
        self.fc1 = nn.Linear(18, 20)
        self.fc2 = nn.Linear(20, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to convert history to feature vector
def history_to_feature(history):
    # Convert history to a binary matrix
    matrix = np.zeros((6, 3))
    for roll in history:
        matrix[roll - 1, :] = 1

    # Flatten matrix to a 1D array and return as PyTorch tensor
    return torch.tensor(matrix.flatten(), dtype=torch.float)


# Load the training data
data = [[3, 1, 1], [6, 5, 2], [4, 4, 4], [6, 3, 1], [6, 2, 1], [5, 3, 2], [5, 4, 3], [6, 4, 3], [6, 4, 2], [5, 3, 2],
        [5, 4, 1], [5, 3, 1], [3, 2, 1], [6, 5, 4], [4, 4, 2], [6, 5, 4], [3, 3, 1], [5, 2, 1], [6, 6, 3], [6, 4, 3],
        [4, 3, 3], [6, 4, 4], [2, 2, 1], [6, 5, 1], [4, 2, 2], [5, 2, 1], [2, 2, 1], [6, 3, 1], [4, 3, 1], [6, 2, 1],
        [4, 4, 1], [5, 1, 1], [6, 5, 3], [5, 2, 1], [6, 6, 4], [4, 3, 3], [3, 2, 2], [5, 5, 2], [6, 6, 2], [6, 3, 2],
        [3, 2, 2], [6, 5, 2], [4, 3, 2], [2, 1, 1], [5, 3, 2], [5, 3, 2], [3, 2, 2], [2, 2, 1], [5, 3, 2], [6, 5, 4],
        [6, 4, 4], [6, 3, 1], [4, 3, 2], [6, 4, 2], [6, 4, 4], [4, 4, 3], [5, 5, 5], [4, 3, 1], [2, 2, 2], [5, 4, 3],
        [6, 1, 1], [5, 2, 2], [2, 1, 1], [4, 2, 2], [6, 5, 1], [5, 4, 1], [6, 6, 1]]
X = []
y = []
for i in range(len(data) - 1):
    X.append(history_to_feature(data[i]))
    y.append(torch.tensor(data[i + 1], dtype=torch.float))

# Convert data to PyTorch tensors
X = torch.stack(X)
y = torch.stack(y)

# Train the model
model = DiceModel()
# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())

# Save the trained model
torch.save(model.state_dict(), '3dice_model.pth')

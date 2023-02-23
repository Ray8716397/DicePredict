import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate training data
num_samples = 10000
roll_history_length = 10  # number of previous rolls to include in input
X_train = np.random.randint(1, 7, size=(num_samples, roll_history_length, 3))
y_train = np.sum(X_train, axis=(1, 2))  # target is sum of three rolls

# Define a PyTorch model with one LSTM layer and one output layer
class SicBoModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SicBoModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = SicBoModel(3, 32, 1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(50):
    running_loss = 0.0
    for i in range(num_samples):
        inputs = torch.from_numpy(X_train[i:i+1, :, :]).float()
        labels = torch.from_numpy(y_train[i:i+1]).float().unsqueeze(-1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/num_samples:.4f}")

# Generate a prediction
roll_history = np.array([[[1, 6, 2], [5, 3, 4], [6, 1, 2], [5, 4, 3], [2, 3, 1], [6, 6, 2], [1, 2, 5], [3, 5, 6], [1, 1, 1], [4, 5, 6]]])  # example history of previous rolls
X_test = torch.from_numpy(roll_history).float()
y_pred = model(X_test)
print("Predicted sum of the next roll:", y_pred.item())

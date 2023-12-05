import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# print(X_numpy)
# print(Y_numpy)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
# print(Y)
Y = Y.view(Y.shape[0], 1)
# print(Y)

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 200
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 2 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')


# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

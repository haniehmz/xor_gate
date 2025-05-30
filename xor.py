import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

np.random.seed(0)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

epochs = 50000
learning_rate = 0.1
error_list = []

for epoch in range(epochs):
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, wout) + bout
    final_output = sigmoid(final_input)

    error = y - final_output
    mse = np.mean(np.square(error))
    error_list.append(mse)

    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    wout += hidden_output.T.dot(d_output) * learning_rate
    bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(error_list)
plt.title('Error Function')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 2, 2)
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
hidden_layer = sigmoid(np.dot(grid, wh) + bh)
output_layer = sigmoid(np.dot(hidden_layer, wout) + bout)
zz = output_layer.reshape(xx.shape)

plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap='coolwarm')
plt.title('Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Final Output:")
print(final_output)
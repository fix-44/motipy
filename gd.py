import numpy as np
import matplotlib.pyplot as plt

# Data sintetik
np.random.seed(42)
x = 2 * np.random.rand(100)
y = 4 + 3 * x + np.random.randn(100)


# Fungsi kehilangan (cost function)
def compute_cost(w, x, y):
    N = len(y)
    return (1 / N) * np.sum((y - (w * x)) ** 2)


# Fungsi untuk melakukan gradient descent
def gradient_descent(x, y, learning_rate=0.01, n_iterations=1000):
    w = 0.0  # Inisialisasi parameter
    cost_history = []

    for i in range(n_iterations):
        # Hitung gradien
        gradient = -(2 / len(y)) * np.sum(x * (y - (w * x)))

        # Update parameter
        w -= learning_rate * gradient

        # Simpan nilai cost untuk visualisasi
        cost = compute_cost(w, x, y)
        cost_history.append(cost)

    return w, cost_history


# Jalankan gradient descent
learning_rate = 0.01
n_iterations = 1000
optimal_w, cost_history = gradient_descent(x, y, learning_rate, n_iterations)

# Tampilkan hasil
print(f'Optimal weight: {optimal_w}')
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Reduction Over Iterations')
plt.show()

# Visualisasi hasil
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, optimal_w * x, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)
X = np.random.rand(1000, 1)
Y = 2 * X + 5 + .2 * np.random.randn(1000, 1)

one = np.ones_like(X)

X_bar = np.concatenate((one, X), axis = 1)
print(X_bar)
Ap = X_bar.T @ X_bar
b = X_bar.T @ Y

w = np.linalg.pinv(Ap) @ b

x0 = np.linspace(0, 1, 2)
y0 = w[0, 0] + w[1, 0] * x0

print(w.T)

plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(X, Y, 'b.')
plt.plot(x0, y0, 'r', linewidth = 1)
plt.show()
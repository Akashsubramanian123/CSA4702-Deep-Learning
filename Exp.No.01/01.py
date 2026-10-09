import numpy as np
import matplotlib.pyplot as plt

# 1. Vector Operations
v1 = np.array([2.0, 3.0])
v2 = np.array([4.0, 1.0])

# Dot product & Norm (magnitude)
dot_product = np.dot(v1, v2)
norm_v1 = np.linalg.norm(v1)

print("--- Vector Operations ---")
print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Dot Product (v1 . v2): {dot_product}")
print(f"L2 Norm of Vector 1: {norm_v1:.4f}\n")

# 2. Matrix Operations
# Matrix Multiplication: Simulated Linear Layer (Y = X @ W + b)
X = np.array([[1.0, 2.0], 
              [3.0, 4.0]])       # Input batch (2 samples, 2 features)

W = np.array([[0.5, -0.2], 
              [0.8,  0.4]])      # Weights (2x2)

b = np.array([0.1, -0.1])        # Bias (1x2)

# Affine transformation & Element-wise ReLU activation
Z = np.matmul(X, W) + b
A = np.maximum(0, Z)             # ReLU

print("--- Matrix Operations (Neural Layer Simulation) ---")
print(f"Input Matrix X:\n{X}")
print(f"Weight Matrix W:\n{W}")
print(f"Linear Output (X @ W + b):\n{Z}")
print(f"Activated Output (ReLU):\n{A}\n")

# 3. Visualization: Vector Addition & Projection
v_sum = v1 + v2

plt.figure(figsize=(7, 7))
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Plot vectors from origin
origin = [0], [0]
plt.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r'$v_1$ (2, 3)')
plt.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', label=r'$v_2$ (4, 1)')
plt.quiver(*origin, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='crimson', label=r'$v_1 + v_2$')

plt.xlim(-1, 8)
plt.ylim(-1, 6)
plt.grid(True, linestyle=':', alpha=0.6)
plt.title("Vector Operations: Addition in 2D Space")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
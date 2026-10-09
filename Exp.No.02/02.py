import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate Synthetic 2D Data with a Correlated Distribution
np.random.seed(42)
mean = [0, 0]
cov_ground_truth = [[3.0, 2.2], 
                    [2.2, 2.0]]

data = np.random.multivariate_normal(mean, cov_ground_truth, size=500)
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])

# 2. Covariance Matrix Computation
# Using Pandas
cov_matrix_pd = df.cov()

# Using NumPy (rowvar=False calculates covariance between columns)
cov_matrix_np = np.cov(df.values, rowvar=False)

print("--- Empirical Covariance Matrix ---")
print(cov_matrix_pd, "\n")

# 3. Eigen Decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_np)

# Sort eigenvectors by eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("--- Eigenvalues & Eigenvectors ---")
for i in range(len(eigenvalues)):
    print(f"Eigenvalue {i+1}: {eigenvalues[i]:.4f}")
    print(f"Eigenvector {i+1}: {eigenvectors[:, i]}\n")

# 4. Visualization of Data Spread and Eigenvectors (Principal Axes)
plt.figure(figsize=(8, 7))
plt.scatter(df['Feature_1'], df['Feature_2'], alpha=0.4, color='royalblue', label='Data Points')

# Mean center
center = df.mean().values

# Plot eigenvectors scaled by standard deviations (sqrt of eigenvalue)
for i in range(len(eigenvalues)):
    scale = np.sqrt(eigenvalues[i]) * 2  # 2 standard deviations
    v = eigenvectors[:, i] * scale
    plt.annotate('', xy=(center[0] + v[0], center[1] + v[1]), xytext=(center[0], center[1]),
                 arrowprops=dict(facecolor='red' if i == 0 else 'orange', 
                                 edgecolor='black', width=2, headwidth=8))
    plt.plot([], [], color='red' if i == 0 else 'orange', 
             label=f'PC {i+1} (λ={eigenvalues[i]:.2f})')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axis('equal')
plt.title("Data Covariance and Principal Directions (Eigenvectors)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
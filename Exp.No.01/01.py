import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [22, 28, 2],
    [35, 55, 5],
    [48, 82, 8],
    [29, 42, 3],
    [41, 68, 6]
], dtype=float)

normalized = (
    (data - data.min(axis=0)) /
    (data.max(axis=0) - data.min(axis=0))
)

print("Original matrix:\n", data)
print("Normalized matrix:\n", np.round(normalized, 3))

plt.imshow(normalized, cmap="viridis", aspect="auto")
plt.colorbar(label="Normalized value")
plt.xticks(range(3), ["Age", "Income", "Purchases"])
plt.yticks(range(5), ["C1", "C2", "C3", "C4", "C5"])
plt.title("Normalized Customer Features")
plt.show()
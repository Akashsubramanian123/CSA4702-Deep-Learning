import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

categories = ["Electronics", "Clothing", "Groceries", "Books"]

observed = np.array([0.40, 0.25, 0.20, 0.15])
predicted = np.array([0.30, 0.30, 0.25, 0.15])

kl = entropy(observed, predicted)

print("Observed distribution:", observed)
print("Predicted distribution:", predicted)
print("KL Divergence:", round(kl, 4), "nats")

x = np.arange(len(categories))
plt.bar(x - 0.18, observed, 0.36, label="Observed")
plt.bar(x + 0.18, predicted, 0.36, label="Predicted")
plt.xticks(x, categories, rotation=15)
plt.ylabel("Probability")
plt.title("Customer Preference Distributions")
plt.legend()
plt.show()
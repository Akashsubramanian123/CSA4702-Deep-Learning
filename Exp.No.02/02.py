import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "Income": [25, 32, 40, 48, 55, 63, 72, 80],
    "Spending": [18, 24, 29, 38, 41, 49, 58, 64],
    "Purchase_Frequency": [2, 3, 3, 5, 5, 6, 8, 8]
})

print("Covariance Matrix:\n", data.cov())
print("\nCorrelation Matrix:\n", data.corr())

plt.imshow(data.corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(3), data.columns, rotation=20)
plt.yticks(range(3), data.columns)
plt.title("Retail Feature Correlation")
plt.show()

plt.scatter(data["Income"], data["Spending"])
plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Income vs Spending")
plt.show()
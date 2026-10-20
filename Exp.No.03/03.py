import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

defects = np.array([2, 1, 3, 0, 2, 4, 1, 3, 2, 2, 5, 1])

lam = defects.mean()
probability = poisson.pmf(3, mu=lam)

print("Average defects:", lam)
print("Probability of exactly 3 defects:", probability)

x = np.arange(0, 9)
plt.bar(x, poisson.pmf(x, mu=lam))
plt.xlabel("Number of defects")
plt.ylabel("Probability")
plt.title("Poisson Distribution of Defects")
plt.show()
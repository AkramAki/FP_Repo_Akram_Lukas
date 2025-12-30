import matplotlib.pyplot as plt
import numpy as np


# gold avg and std
gold_heights = np.array([4.14, 4.86, 8.25, 5.21, 6.75, 11.55])
gold_avg = np.mean(gold_heights)
gold_std = np.std(gold_heights)

# print results
print(f"Gold Average Height: {gold_avg:.2f} nm")
print(f"Gold Standard Deviation: {gold_std:.2f} nm")
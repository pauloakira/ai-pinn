import matplotlib.pyplot as plt
import numpy as np

# Hardcoded data
elements = [24, 48, 96, 192, 384]
order4_mean = [0.0002946, 0.00030283, 0.00019242, 0.0000445, 0.00111329]
order4_std = [1.35e-8, 6.74e-5, 2.99e-5, 8.39e-8, 2.46e-5]
order5_mean = [0.00032217, 0.00029875, 0.00019449, 0.0000614, 0.00111669]
order5_std = [7.11e-5, 6.82e-5, 4.92e-5, 2.71e-5, 3.74e-5]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot with error bars
plt.errorbar(elements, order4_mean, yerr=order4_std, marker='o', label='Order 4', capsize=5)
plt.errorbar(elements, order5_mean, yerr=order5_std, marker='s', label='Order 5', capsize=5)

# Customize the plot
plt.xlabel('Elements')
plt.ylabel('Mean')
plt.title(r'$H^1$-error comparison of order 4 and order 5 with standard deviation')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Use logarithmic scale if needed
plt.yscale('log')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
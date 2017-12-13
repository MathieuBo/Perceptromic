import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# Load RRHO matrix obtained from R
rrho = np.loadtxt('../../var_combination/results/rrho_matrix.txt')

# Prepare the heatmap figure.
# The origin has to be in the lower left corner to fit with the classical picture of RRHO
plt.figure()
plt.imshow(rrho, origin='lower', cmap='jet', interpolation='bicubic')
plt.colorbar(label='-log(p value)')
plt.title('Rank-Rank hypergeometric plot: LB vs noLB')
plt.xticks([])
plt.yticks([])
plt.savefig('../../var_combination/results/rrho_plot.pdf')
plt.show()

# Selection of the diagonal as a bin to bin comparison
diag = np.diag(rrho)

# Smoothening of the curve
x_new = np.linspace(0, len(diag), 200)
smoothed_curve = spline(np.arange(len(diag)), diag, x_new)

# Plot
fig, ax = plt.subplots(figsize=(20, 3))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(np.arange(len(diag)), diag, '.k')
plt.plot(x_new, smoothed_curve, '--k')

plt.ylabel('-log(p value)')
plt.xticks([])
plt.xlim((0, len(diag)))
plt.ylim((0, int(np.max(diag))+1))
plt.tight_layout()
plt.savefig('../../var_combination/results/rrho_diag.pdf')
plt.show()
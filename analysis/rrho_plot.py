import numpy as np
import matplotlib.pyplot as plt

rrho = np.loadtxt('../../var_combination/results/rrho_matrix.txt')

plt.figure()
plt.imshow(rrho, origin='lower', cmap='jet', interpolation='bicubic')
plt.colorbar(label='-log(p value)')
plt.title('Rank-Rank hypergeometric plot: LB vs noLB')
plt.xticks([])
plt.yticks([])
plt.savefig('../../var_combination/results/rrho_plot.pdf')
plt.show()
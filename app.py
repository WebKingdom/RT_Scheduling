import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# TODO
# Fixing random state for reproducibility
np.random.seed(19680801)

# Z = np.random.rand(6, 10)

Z = [[1, 2, 3],
     [4, 8, 10]]

fig, (ax0, ax1) = plt.subplots(2, 1)

c = ax0.pcolor(Z)
ax0.set_title('default: no edges')

c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')

fig.tight_layout()
plt.show()

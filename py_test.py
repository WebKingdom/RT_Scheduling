import matplotlib.pyplot as plt
import numpy as np

msg = "Hi Soma"
print(msg)

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()

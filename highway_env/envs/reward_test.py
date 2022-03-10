import numpy as np
import matplotlib.pyplot as plt



distances = np.linspace(0,0.8,50)


plt.plot(distances,np.exp(-20.5*distances**2),label="40.5")
plt.plot(distances,np.exp(-2.5*distances**2),label="2.5")
plt.plot(distances,np.exp(-6.5*distances**2),label="6.5")
plt.plot(distances,np.exp(-7.5*distances**2),label="7.5")
plt.plot(distances,np.exp(-1.5*distances**2),label="1.5")
plt.legend()
plt.show()
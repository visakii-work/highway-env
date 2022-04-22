import numpy as np
import matplotlib.pyplot as plt



distances = np.linspace(0,0.8,50)

speeds = np.linspace(0,12,12)


plt.plot(speeds/12)
plt.plot(np.exp(-0.06*(speeds-12)**2),label="0.06")
plt.plot(np.exp(-0.04*(speeds-12)**2),label="0.04")
plt.plot(np.exp(-0.02*(speeds-12)**2),label="0.02")
plt.legend()
plt.figure()



plt.plot(distances,np.exp(-15.5*distances),label="40.5")
plt.plot(distances,np.exp(-2.5*distances),label="2.5")
plt.plot(distances,np.exp(-5.5*distances),label="5.5")
plt.plot(distances,np.exp(-7.5*distances),label="7.5")
plt.plot(distances,np.exp(-6.5*distances),label="6.5")
plt.plot(distances,1-(distances)**2,label="1-d sq")
plt.plot(distances,1-(2*distances)**2,label="1-d sq")
plt.legend()
plt.show()
from dataClean import dataset
import matplotlib.pyplot as plt


dataset.info()

ax1 = plt.subplot(1,1,1)
ax1.scatter(dataset['LCD'], dataset['Heat_furfural'],c="r")
plt.show()
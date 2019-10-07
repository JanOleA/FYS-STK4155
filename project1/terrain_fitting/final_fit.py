import sys
sys.path.insert(1, '../resources/')

from regression import *
from resources import *
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

sns.set()
terrain = imread("terrain_data/SRTM_data_Norway_1.tif")
print("Original dataset size:", terrain.shape)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)

x_start = 500
y_start = 1500

plt.figure()
alt = plt.imshow(terrain, cmap="gray") # show entire loaded terrain

c_terrain = terrain[y_start:y_start + N, x_start:x_start + N] # crop terrain to NxN square
plt.figure()
alt = plt.imshow(c_terrain, cmap="gray") # show cropped terrain

scale_constant = np.max(c_terrain)
c_terrain = c_terrain/scale_constant # normalize input data
plt.title("Squared terrain, normalized")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

m_list = [2, 6, 10, 20]

plot_surface(x_m, y_m, c_terrain)
plt.title("Data")
plt.figure()
plt.imshow(c_terrain, cmap="gray")
plt.title("Data")

for m in m_list:
    print(m)
    X = design_matrix(x_m, y_m, m)
    model = OLS(X, c_terrain)
    z_predict = model(X).reshape((N,N))


    plot_surface(x_m, y_m, z_predict)
    plt.title(str("m = ") + str(m))
    plt.figure()
    plt.imshow(z_predict, cmap="gray")
    plt.title(str("m = ") + str(m))

plt.show()

import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/home/mbuenabad/Documents/codes/packages/physics/class/class_adam-3.1/output/adam/adam_01_background.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['adam_01_background']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['rho_g_twin', 'rho_ur_twin', 'rho_b_twin']
tex_names = ['(8\\pi G/3)rho_g_twin', '(8\\pi G/3)rho_ur_twin', '(8\\pi G/3)rho_b_twin']
x_axis = 'z'
ylim = []
xlim = []
ax.loglog(curve[:, 0], abs(curve[:, 16]))
ax.loglog(curve[:, 0], abs(curve[:, 17]))
ax.loglog(curve[:, 0], abs(curve[:, 18]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('z', fontsize=16)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/home/mbuenabad/Documents/codes/packages/physics/class/class_adam-3.1/output/adam/adam_01_perturbations_k4_s.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['adam_01_perturbations_k4_s']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['theta_cdm', 'theta_b', 'theta_g', 'theta_idm_dr', 'theta_idr']
tex_names = ['theta_g', 'theta_b', 'theta_idr', 'theta_idm_dr', 'theta_cdm']
x_axis = 'tau [Mpc]'
ylim = []
xlim = []
ax.loglog(curve[:, 0], abs(curve[:, 20]))
ax.loglog(curve[:, 0], abs(curve[:, 9]))
ax.loglog(curve[:, 0], abs(curve[:, 3]))
ax.loglog(curve[:, 0], abs(curve[:, 18]))
ax.loglog(curve[:, 0], abs(curve[:, 16]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('tau [Mpc]', fontsize=16)
plt.show()
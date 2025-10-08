from emulator_lib import eigenvector_continuation as evc
from emulator_lib import lec_values as lecs
from emulator_lib import aux as aux
import numpy as np
import matplotlib.pyplot as plt
import sys

parameters = lecs.NNLOsat_parameter_names
#constant first
parameters.insert(0,'const')
LECvalues = lecs.NNLOsat_LECs

path = './../cc_output/spcc64_o16_nnlosat_emax6_hw18/spcc_matrices/'
file_base_H = 'hbar_20percent64_%s_nnlosat_mass_16_N06E16_hw16_OSC.dat'
file_norm = 'norm_20percent64_cE_nnlosat_mass_16_N06E16_hw16_OSC.dat'
files_obs = ['eccentricity_20percent64_c1_nnlosat_mass_16_N06E16_hw16_OSC.dat']
names_obs = ['E']

subspace_dim = 64
domain_dim = 16

em_16O = evc.read_emulator(name='em_16O', domain_dimension=domain_dim, subspace_dimension=subspace_dim, parameter_list=parameters,
                           path = path, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True, files_obs = files_obs, names_obs = names_obs)

em = em_16O

# test-point to reproduce NNLOsat result using emulator
test_point = []
for idx, par in enumerate(parameters):
    try:
        test_point.append(LECvalues[par])
    except KeyError:
        continue



print(f'test_point = {test_point}')
obs, eigvec_O, spectrum = em.evaluate(test_point,level=0)

print(f'eccentricity = {obs}')
print(f'energy       = {spectrum[0]}')


# ordering in xval file: C3S1 C3P2 C1S0 Ct1S0pp Ct1S0nn C3P0 C3P1 Ct1S0np cE CE1 Ct3S1 c3 c1 cD C1P1 c4
xval_lec_values = './../cc_input/cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20_percent_200_points/list_of_points_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20percent_200_points.txt'
xval_points, names_used = aux.read_list_of_points(xval_lec_values, order_parameters=parameters, return_names=True)
print("Columns returned (in order):", names_used)
print("xval_points shape:", xval_points.shape)

# some exact calucations fail. They are commented out.
xval_exact_values = './../cc_input/cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20_percent_200_points/o16_xval_200_pnts.txt'
idx, cc_vals_energy, cc_vals_eccentricity = np.loadtxt(xval_exact_values,comments="#", usecols=(0, 1, 2), unpack=True)

# remove the commented out xval lec values read above. only the ones with index idx_keep are kept
idx_keep = np.asarray(idx, dtype=int)
idx_keep = np.unique(idx_keep)  # remove duplicates if any, keeps sorted order

# Apply mask
xval_points = xval_points[idx_keep, :]

print(f'found {xval_points.shape[0]} LEC values and {cc_vals_energy.shape[0]} CC-energies and {cc_vals_eccentricity.shape[0]} eccentricites')


spcc_vals_energy = []
spcc_vals_eccentricity = []

for idx, LECvalue in enumerate(xval_points):

    print(idx)
    obs, eigvec_O, spectrum = em.evaluate(LECvalue,level=0)

    print(f' SPCC energy = {spectrum[0].real} | SPCC eccentricity = {obs[0]}')
    print(f'   CC energy = {cc_vals_energy[idx]} |   CC eccentricity = {cc_vals_eccentricity[idx]}')
    spcc_vals_eccentricity.append(obs[0])
    spcc_vals_energy.append(spectrum[0].real)


# --- ENERGY PLOT ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(cc_vals_energy, spcc_vals_energy, s=30, alpha=0.8)
xy = np.linspace(*ax.get_xlim())
ax.plot(xy, xy, color='k', lw=1)
ax.set_xlabel('Exact')
ax.set_ylabel('Emulator')
ax.set_title('Energy')
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('xval_energy.pdf')
plt.show()

# --- ECCENTRICITY PLOT ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(cc_vals_eccentricity, spcc_vals_eccentricity, s=30, alpha=0.8)
xy = np.linspace(*ax.get_xlim())
ax.plot(xy, xy, color='k', lw=1)
ax.set_xlabel('Exact')
ax.set_ylabel('Emulator')
ax.set_title('Eccentricity')
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('xval_eccentricity.pdf')
plt.show()

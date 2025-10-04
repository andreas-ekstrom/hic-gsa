import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import h5py
import re
import pandas as pd
import fig_preamble as pre
from emulators import lec_values as lecs
from emulators import eigenvector_continuation as evc
from scipy.stats import norm

def find_nearest(array, value):
    array = np.asarray(array)
    diff_arrays = np.abs(array - value)
    idx = (diff_arrays.argmin())
    idxs = np.where(diff_arrays == diff_arrays.min())[0]
    return idx, array[idx], idxs
           
def read_list_of_points(filename, skip_first = True):    
    points = []
    with open(filename,'r') as f:
        if skip_first:
            next(f)
        for line in f:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            points.append(myarray)
    return np.array(points)

def read_list_of_indexed_points(filename, skip_first = True):    
    num_lines = sum(1 for line in open(filename))
    if skip_first:
        num_lines -= 1

    #print('num_lines:',num_lines)
    points = [None] * num_lines
    
    with open(filename,'r') as f:
        if skip_first:
            next(f)
        for line in f:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            points[int(myarray[0])] = myarray[1]
        return np.array(points)

from pathlib import Path
base_path = Path(__file__).parent

import time

np.random.seed(1234)

parameters = lecs.NNLOsat_parameter_names
parameters.insert(0,'const')
print(parameters)

LECvalues = lecs.NNLOsat_LECs

zero_point = []
for idx, par in enumerate(parameters):
    try:
        zero_point.append(LECvalues[par])
    except KeyError:
        continue

# SP-CCSDT emulator

ddim = 16
sdim = 64
spcc = evc.read_emulator(name='O16', domain_dimension=ddim, subspace_dimension=sdim, parameter_list=parameters,
                         path='./../../cc_output/spcc64_o16_nnlosat_emax6_hw18/spcc_matrices/',
                         file_base_H='hbar_20percent64_%s_nnlosat_mass_16_N06E16_hw16_OSC.dat',
                         file_norm='norm_20percent64_c1_nnlosat_mass_16_N06E16_hw16_OSC.dat',
                         remove_constant=True,
                         files_obs=['eccentricity_20percent64_c1_nnlosat_mass_16_N06E16_hw16_OSC.dat'],
                         names_obs=['Eccentricity'])

training_lecs = read_list_of_points('../../cc_input/cc_input/cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20_percent_200_points/list_of_points_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20percent_200_points.txt')

cc_energies = read_list_of_indexed_points('../../cc_input/cc_input/cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_20_percent_200_points/o16_xval_200_pnts.txt',skip_first=False)

print('zero-point')
print(spcc.evaluate(zero_point)[0])

#print('zero-point SAMPLE SELECT')
#stable_sample = spcc.sample_select(zero_point)
#print(spcc.evaluate(zero_point, target=stable_sample)[0])

zero_point_spectrum = spcc.evaluate(zero_point)[2]
for idx, pnt in enumerate(zero_point_spectrum):
    print(idx, pnt)

print('=======================================================================')
print('=======================================================================')

print('Selecting best match in training spectrum. Idx indicates select spcc-state')

print('vec \t idx \t R(spcc) \t I(spcc) \t train-val \t diff')
best_val = []
for vec_idx, training_vector in enumerate(training_lecs):
    nearest_state = 0
    E_spectrum = spcc.evaluate(training_vector)[2]
    spcc_value = spcc.evaluate(training_vector)[0]
    state, spcc_value, states = find_nearest(E_spectrum, cc_energies[vec_idx])
    best_val.append(spcc_value.real)

    # check obs/radii for nearest state
    #spcc_value_state = spcc.evaluate(training_vector, level=[state])[0]
    #print('Rp2', spcc_value_state[1], cc_Rp2[vec_idx], spcc_value_state[1] - cc_Rp2[vec_idx])
    #print('Rn2', spcc_value_state[2], cc_Rn2[vec_idx], spcc_value_state[2] - cc_Rn2[vec_idx])

    if len(states) > 1:
        print('WARNING: more than one best state!!')
    print('%d \t %d \t %.6f \t i%.6f \t %.6f \t %.6f'%(vec_idx, state,spcc_value.real,spcc_value.imag,cc_energies[vec_idx],spcc_value.real-cc_energies[vec_idx]))

print('=======================================================================')
print('=======================================================================')


# truncate training data and or output

Ecut_lo = -300.0
Ecut_hi = -10.0

LEC_lim = 'Ct3S1'
percent_lim = 20.20

LEC_idx   = list(LECvalues.keys()).index(LEC_lim)
zero_point_value = LECvalues[LEC_lim]
lim = abs((percent_lim/100.0) * zero_point_value)
LEC_lo = zero_point_value - lim
LEC_hi = zero_point_value + lim

print('LEC_lo:',LEC_lo)
print('LEC_hi:',LEC_hi)

# loop through training vectors and output
truncated_training_vectors = []
truncated_cc_energies = []
dropped_vectors = []
kept_vectors = []
for idx,vec in enumerate(training_lecs):
    #print(idx)
    keep_idx_LEC = False
    keep_idx_energy = False
    #print(vec[LEC_idx], cc_energies[idx])
    if LEC_lo < vec[LEC_idx] < LEC_hi:
        keep_idx_LEC = True
        #print('LEC keep')
    if Ecut_lo < cc_energies[idx] < Ecut_hi:
        keep_idx_energy = True
        #print('E keep')
    if keep_idx_LEC and keep_idx_energy:
        kept_vectors.append(idx)
    else:
        dropped_vectors.append(idx)

print('Based on Ecut and LEC cut, the following vectors should be removed:')
print(dropped_vectors)
print('Based on Ecut and LEC cut, the following vectors should be kept:')
print(kept_vectors)

print('-> zero-point')
print(spcc.evaluate(zero_point,drop_states = dropped_vectors)[0])

zero_point_spectrum = spcc.evaluate(zero_point,drop_states = dropped_vectors)[2]
for idx, pnt in enumerate(zero_point_spectrum):
    print(idx, pnt)

print('=======================================================================')
print('=======================================================================')

print('re-running check on training vectors')
print('idx \t dropped \t R(spcc) \t I(spcc) \t train-val \t diff')
best_val = []
for vec_idx, training_vector in enumerate(training_lecs):

    drop = 'no'
    if vec_idx in dropped_vectors:
        drop = 'yes'
        
    nearest_state = 0
    E_spectrum = spcc.evaluate(training_vector, drop_states = dropped_vectors)[2]
    spcc_value= spcc.evaluate(training_vector,drop_states = dropped_vectors)[0]
    state, spcc_value, states = find_nearest(E_spectrum, cc_energies[vec_idx])
    best_val.append(spcc_value.real)
    
    if len(states) > 1:
        print('WARNING: more than one best state!!')
    print('%d \t %s \t %.6f \t i%.6f \t %.6f \t %.6f'%(state,drop, spcc_value.real,spcc_value.imag,cc_energies[vec_idx],spcc_value.real-cc_energies[vec_idx]))

print('=======================================================================')
print('=======================================================================')

#xval_lecs = np.delete(training_lecs, kept_vectors, axis=0)
#xval_energies = np.delete(cc_energies, kept_vectors, axis=0)

#xval_Rp2 =np.delete(Rp2_68, kept_vectors, axis=0) 
#xval_Rn2 =np.delete(Rn2_68, kept_vectors, axis=0)

# add more data from other cc-runs
training_lecs_68 = read_list_of_points('/Users/andreas/Documents/manuscripts/Pb208_spcc/cc_input/cc_lhs_input_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10_percent_68_points/list_of_points_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10percent_68_points.txt')
#
cc_energies_68 = read_list_of_indexed_points('/Users/andreas/Documents/manuscripts/Pb208_spcc/cc_input/cc_lhs_input_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10_percent_68_points/ccsd_energies.txt',skip_first=False)

Rp2_68 = read_list_of_indexed_points('/Users/andreas/Documents/manuscripts/Pb208_spcc/cc_output/cross_validation_data/cc_Rp2.txt',skip_first=False)
Rn2_68 = read_list_of_indexed_points('/Users/andreas/Documents/manuscripts/Pb208_spcc/cc_output/cross_validation_data/cc_Rn2.txt',skip_first=False)

#xval_lecs = np.append(xval_lecs, training_lecs_68, axis=0)
#xval_energies = np.append(xval_energies, cc_energies_68, axis=0)

xval_lecs = training_lecs_68
xval_energies = cc_energies_68
xval_Rp2 = Rp2_68
xval_Rn2 = Rn2_68

print('re-running check on xval vectors')
print('idx \t R(spcc) \t I(spcc) \t train-val \t diff')
best_val = []
for vec_idx, training_vector in enumerate(xval_lecs):

    nearest_state = 0
    E_spectrum = spcc.evaluate(training_vector, drop_states = dropped_vectors)[2]
    spcc_value= spcc.evaluate(training_vector,drop_states = dropped_vectors)[0]
    state, spcc_value, states = find_nearest(E_spectrum, xval_energies[vec_idx])
    best_val.append(spcc_value.real)
    
    if len(states) > 1:
        print('WARNING: more than one best state!!')
    print('%d \t %.6f \t i%.6f \t %.6f \t %.6f'%(state, spcc_value.real,spcc_value.imag,xval_energies[vec_idx],spcc_value.real-xval_energies[vec_idx]))

print('=======================================================================')
print('=======================================================================')

nxval = len(xval_Rn2)

spcc_output = spcc.batch_evaluate(xval_lecs, drop_states = dropped_vectors)
#spcc_output = spcc.batch_evaluate(xval_lecs, use_best_sample = True)

print('len(spcc_output):',len(spcc_output))
# remove xval energies outside of energy cut:
Exval_lo = -1700.0
Exval_hi = -800.0

xval_delete = []
for idx, output in enumerate(spcc_output):
    if Exval_lo < output[0] < Exval_hi:
        pass
    else:
        xval_delete.append(idx)
        #print(output[0],np.sqrt(output[1]),np.sqrt(output[2]))

spcc_output   = np.delete(spcc_output, xval_delete, axis=0)
xval_energies = np.delete(xval_energies, xval_delete, axis=0)
xval_Rp2      = np.delete(xval_Rp2, xval_delete, axis=0)
xval_Rn2      = np.delete(xval_Rn2, xval_delete, axis=0)

print('deleted xval entries:')
print(xval_delete)

print('=======================================================================')
print('=======================================================================')

spcc_delta_go = spcc.evaluate(zero_point,drop_states = dropped_vectors)[0]
cc_delta_go = [-1381.44472546,28.25217438,29.67949486] 

fig_size = pre.figure_article(rows=1.4,columns=1)

ms=18

fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax1.set_title('$^{208}$Pb energy (MeV)')
ax1.scatter(xval_energies,spcc_output[:,0],marker='s', s=ms, facecolor = pre.Ecol, edgecolor = 'black', linewidths=0.05, label=r'SPCC')
ax1.scatter(cc_delta_go[0],spcc_delta_go[0], marker='s',s=70, zorder=100, color='black', facecolor='black', linewidth=0.5, label=r'$\Delta$-NNLO$_{\rm go}$(394)')
ax1.set_xlabel('CCSD')
ax1.set_ylabel('SPCC')
ax1.axvline(-1636.43,color='black',linestyle='--')
ax1.set_xticks(np.arange(-1600, -700, 200.0))
plt.legend()
x = np.linspace(*ax1.get_xlim())
ax1.plot(x, x, color='k')
plt.tight_layout()
plt.savefig('xval_E.pdf')

fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax2.set_title('$^{208}$Pb point-proton radius (fm)')
ax2.scatter(np.sqrt(xval_Rp2),np.sqrt(spcc_output[:,1]),marker='s', s=ms, facecolor = pre.Rpcol, edgecolor='black',linewidths=0.05, label=r'SPCC')
ax2.scatter(np.sqrt(cc_delta_go[1]),np.sqrt(spcc_delta_go[1]), marker='s',s=70, zorder=100, color='black', facecolor='black', linewidth=0.5, label=r'$\Delta$-NNLO$_{\rm go}$(394)')
ax2.set_xlabel('CCSD')
ax2.set_ylabel('SPCC')
plt.legend()
x = np.linspace(*ax2.get_xlim())
ax2.plot(x, x, color='k')
plt.tight_layout()
plt.savefig('xval_Rp.pdf')

fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax3.set_title('$^{208}$Pb point-neutron radius (fm)')
ax3.scatter(np.sqrt(xval_Rn2),np.sqrt(spcc_output[:,2]),marker='s', s=ms, facecolor = pre.Rncol, edgecolor='black',linewidths=0.05, label=r'SPCC')
ax3.scatter(np.sqrt(cc_delta_go[2]),np.sqrt(spcc_delta_go[2]), marker='s',s=70, zorder=100, color='black', facecolor='black', linewidth=0.5, label=r'$\Delta$-NNLO$_{\rm go}$(394)')
ax3.set_xlabel('CCSD')
ax3.set_ylabel('SPCC')
plt.legend()
x = np.linspace(*ax3.get_xlim())
ax3.plot(x, x, color='k')
plt.tight_layout()
plt.savefig('xval_Rn.pdf')

fig4, ax4 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax4.set_title('$^{208}$Pb point neutron-skin thickness (fm)')
ax4.scatter(np.sqrt(xval_Rn2)-np.sqrt(xval_Rp2),np.sqrt(spcc_output[:,2])-np.sqrt(spcc_output[:,1]),marker='s', s=ms, facecolor = pre.Rscol, edgecolor='black',linewidths=0.05, label=r'SPCC')
ax4.scatter(np.sqrt(cc_delta_go[2]) - np.sqrt(cc_delta_go[1]),np.sqrt(spcc_delta_go[2]) - np.sqrt(spcc_delta_go[1]), marker='s',s=70, zorder=100, color='black', facecolor='black', linewidth=0.5, label=r'$\Delta$-NNLO$_{\rm go}$(394)')
ax4.set_xlabel('CCSD')
ax4.set_ylabel('SPCC')
plt.legend()
x = np.linspace(*ax4.get_xlim())
ax4.plot(x, x, color='k')
plt.tight_layout()
plt.savefig('xval_Rs.pdf')

fig5, ax5 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax5.set_title('$^{208}$Pb Residuals point neutron-skin thickness (fm)')
diffs = (np.sqrt(xval_Rn2)-np.sqrt(xval_Rp2)) - (np.sqrt(spcc_output[:,2])-np.sqrt(spcc_output[:,1]))
sRs = np.std(diffs)
mRs = np.mean(diffs)
print('  mean[Rskin] :',mRs)
print('  sigma[Rskin]:',sRs)
print('3*sigma[Rskin]:',3*sRs)

ax5.hist(diffs,bins=10, color=pre.Rscol, range=[-0.03,0.03])
plt.tight_layout()
plt.savefig('xval_Rs_residuals.pdf')

fig6, ax6 = plt.subplots(ncols=1, nrows=1, figsize = [fig_size[0],fig_size[1]])
ax6.set_title('$^{208}$Pb Residuals energy (MeV)')
diffs = xval_energies-spcc_output[:,0]
sE = np.std(diffs)
mE = np.mean(diffs)
print('  mean[Energy] :',mE)
print('  sigma[Energy]:',sE)
print('3*sigma[Energy]:',3*sE)
ax6.hist(diffs,bins=10, color=pre.Ecol, range=[-10.0,10.0])
plt.tight_layout()
plt.savefig('xval_E_residuals.pdf')

plt.show()


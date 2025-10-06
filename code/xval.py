from emulator_lib import eigenvector_continuation as evc
from emulator_lib import lec_values as lecs
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
files_obs = ['eccentricity_20percent64_cE_nnlosat_mass_16_N06E16_hw16_OSC.dat']
names_obs = ['E']

subspace_dim = 64
domain_dim = 16

em_16O = evc.read_emulator(name='em_16O', domain_dimension=domain_dim, subspace_dimension=subspace_dim, parameter_list=parameters,
                           path = path, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True, files_obs = files_obs, names_obs = names_obs)

em = em_16O

# test-point to reproduce NNLOsat result using emulator
zero_point = []
for idx, par in enumerate(parameters):
    try:
        zero_point.append(LECvalues[par])
    except KeyError:
        continue



print(zero_point)
obs, eigvec_O, spectrum = em.evaluate(zero_point,level=0)

print(obs)
print(spectrum[0])

sys.exit(-1)


            
def read_list_of_points(filename, skip_first = True):    
    points = []
    with open(filename,'r') as f:
        if skip_first:
            next(f)
        for line in f:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            points.append(myarray)
    return np.array(points)

# c1 c2 c3 c4 Ct1S0np Ct1S0nn Ct1S0pp Ct3S1 C1S0 C3P0 C1P1 C3P1 C3S1 CE1 C3P2 cD cE
#xval_points=read_list_of_points('./../spcc_data/ne20_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/list_of_points_Delta_go_394_mass_20_hw_14_10percent_400_points.txt')
#cc_vals = np.loadtxt('./../spcc_data/ne20_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/ne20_HFPAV_400ptns_10percent.txt')

#xval_points=read_list_of_points('./../spcc_data/ne20_HFPAV_400pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14/list_of_points_Delta_go_394_mass_20_hw_14_OPE_vary_20percent_400_points.txt')
#cc_vals = np.loadtxt('./../spcc_data/ne20_HFPAV_400pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14/ne20_HFPAV_400ptns_20percent.txt')

xval_points=read_list_of_points('./../spcc_data/mg24_HFPAV_400pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14/list_of_points_Delta_go_394_mass_20_hw_14_OPE_vary_20percent_400_points.txt')
cc_vals = np.loadtxt('./../spcc_data/mg24_HFPAV_400pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14/mg24_HFPAV_400ptns_20percent.txt')

UHF = []
Egs = []
Ex1_ev = []
Ex2_ev = []

Ex1_ov = []
Ex2_ov = []
E1_ov = []
E2_ov = []

UHF_exact = []
Egs_exact = []
Ex1_exact = []
Ex2_exact = []
E1_exact = []
E2_exact = []

these_states = []
#these_states = [item for item in range(0, 34)]
#print(these_states)

index_list = list(cc_vals[:,0])
index_list = [int(x) for x in index_list]

for idx, LECvalue in enumerate(xval_points):
    try :
        array_index = index_list.index(idx)
    except ValueError:
        print(f'{idx} not in cc-exact values')
        continue

    print(idx)
    idx, E_unpro, E_0, E_2, E_4 = cc_vals[array_index,:]
    print(idx, E_unpro, E_0, E_2, E_4)
    
    HF_unpro_eigval, HF_unpro_eigvec, _,H_unpro, unpro_norm,no0b = em_20Ne_UHF.evaluate(hamiltonian_type = 'non-symmetric', domain_point = LECvalue, drop_states = these_states)
    HF_PAV_0_eigval, HF_PAV_0_eigvec, _, H_PAV_0, PAV_0_norm,no0b = em_20Ne_0.evaluate(hamiltonian_type = 'non-symmetric', domain_point = LECvalue, drop_states = these_states)
    HF_PAV_2_eigval, HF_PAV_2_eigvec, _, H_PAV_2, PAV_2_norm,no0b = em_20Ne_2.evaluate(hamiltonian_type = 'non-symmetric', domain_point = LECvalue, drop_states = these_states)
    HF_PAV_4_eigval, HF_PAV_4_eigvec, _, H_PAV_4,PAV_4_norm,no0b = em_20Ne_4.evaluate(hamiltonian_type = 'non-symmetric', domain_point = LECvalue, drop_states = these_states)

    if HF_PAV_0_eigval >0:
        continue
    if HF_PAV_0_eigval <-300:
        continue
    
    NN = HF_unpro_eigvec.T@unpro_norm@HF_unpro_eigvec
    HH = HF_unpro_eigvec.T@H_unpro@HF_unpro_eigvec
    HF_unpro_overlap = HH/NN + no0b


    NN = HF_unpro_eigvec.T@PAV_0_norm@HF_unpro_eigvec
    HH = HF_unpro_eigvec.T@H_PAV_0@HF_unpro_eigvec
    HF_PAV_0_overlap = HH/NN + no0b

    NN = HF_unpro_eigvec.T@PAV_2_norm@HF_unpro_eigvec
    HH = HF_unpro_eigvec.T@H_PAV_2@HF_unpro_eigvec
    HF_PAV_2_overlap = HH/NN + no0b

    NN = HF_unpro_eigvec.T@PAV_4_norm@HF_unpro_eigvec
    HH = HF_unpro_eigvec.T@H_PAV_4@HF_unpro_eigvec
    HF_PAV_4_overlap = HH/NN + no0b

    if (HF_PAV_2_eigval-HF_PAV_0_eigval) <0:
        continue

    if ((HF_PAV_4_overlap-HF_PAV_0_overlap)/(HF_PAV_2_eigval-HF_PAV_0_eigval)) <1:
        continue    
    
    print(f'E(2+) = {HF_PAV_2_eigval-HF_PAV_0_eigval}')
    
    UHF.append(HF_unpro_eigval)
    UHF_exact.append(E_unpro)

    Egs.append(HF_PAV_0_eigval)
    Egs_exact.append(E_0)
    
    Ex1_ev.append(HF_PAV_2_eigval-HF_PAV_0_eigval)
    E1_ov.append(HF_PAV_2_overlap)
    Ex1_ov.append(HF_PAV_2_overlap-HF_PAV_0_overlap)
    E1_exact.append(E_2)
    Ex1_exact.append(E_2 - E_0)
    
    Ex2_ev.append(HF_PAV_4_eigval-HF_PAV_0_eigval)
    E2_ov.append(HF_PAV_4_overlap)
    Ex2_ov.append(HF_PAV_4_overlap-HF_PAV_0_overlap)
    E2_exact.append(E_4)
    Ex2_exact.append(E_4-E_0)

    R42_exact = (E_4-E_0)/(E_2-E_0)
    R42_emul =  (HF_PAV_4_overlap-HF_PAV_0_overlap)/(HF_PAV_2_overlap-HF_PAV_0_overlap)

    if (R42_exact/R42_emul)<0.95:
        print(f'\n')
        print(f'bad emulator point')
        print(f'{R42_exact} {R42_emul}')
        print(LECvalue)
        print(f'\n')
    if (R42_exact/R42_emul)>1.05:
        print(f'\n')
        print(f'bad emulator point')
        print(f'{R42_exact} {R42_emul}')
        print(LECvalue)
        print(f'\n')        

f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8, 8))

#c = ax.scatter(Egs_exact,Egs,c=Egs,alpha=0.6)
#c = ax.scatter(E1_exact,E1_ov,c=Egs,alpha=0.6)
#c = ax.scatter(E2_exact,E2_ov,c=Egs,alpha=0.6)
#c = ax.scatter(Ex1_exact,Ex1_ov,c=Egs,alpha=0.6)
#c = ax.scatter(Ex2_exact,Ex2_ov,c=Egs,alpha=0.6)
c = ax.scatter(np.array(Ex2_exact)/np.array(Ex1_exact),np.array(Ex2_ov)/np.array(Ex1_ov),c=Egs,alpha=0.6)
cbar = plt.colorbar(c)
cbar.set_label('gs', rotation=270)
xy = np.linspace(*ax.get_xlim())
ax.plot(xy, xy, color='k',lw=1)
ax.set_xlabel('Exact');
ax.set_ylabel('Emulator');

plt.title('Ratio: Excitation energy 4+/2+')
#plt.title('Excitation energy 4+')
# to limit the plot
#plt.ylim(2,5)
#plt.xlim(2,5)

#plt.savefig('Excited_4plus_zoom.pdf')
#plt.savefig('Exc_4_to_Exc_2_zoom.pdf')
plt.savefig('R42.pdf')
#plt.savefig('HF_unprojected.pdf')
plt.show()

##plt.plot(values,UHF,label='EVC (HF)')
##plt.plot(values,Egs,label='EVC (0+)')
##plt.plot(values,Ex1_ev,label='EVC (2+) eigval')
##plt.plot(values,Ex2_ev,label='EVC (4+) eigval')
##plt.plot(values,Ex1_ov,label='EVC (2+) overlap',ls='--',c='blue')
##plt.plot(values,Ex2_ov,label='EVC (4+) overlap',ls='--',c='orange')
#
#plt.plot(values,np.array(Ex2_ov)/np.array(Ex1_ov),label='EVC (4+)')
#
##get exact values
#exact = np.loadtxt('./../spcc_data/ne20_34pts_20_percent_domain_no_isobreak_N10E16_hw14/hf_evc_pav_3nf.txt', comments='#')
##plt.scatter(exact[:,0],exact[:,1],label='exact (0+)')
##plt.scatter(exact[:,0],(exact[:,3]-exact[:,2]),label='exact (2+)')
##plt.scatter(exact[:,0],(exact[:,4]-exact[:,2]),label='exact (4+)')
#plt.scatter(exact[:,0],np.array(exact[:,4]-exact[:,2])/np.array(exact[:,3]-exact[:,2]),label='exact (4+)')
#
#plt.legend()
#plt.savefig('eigval_vs_overlap.pdf')
#
#plt.show()




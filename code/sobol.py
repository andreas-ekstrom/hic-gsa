import numpy as np
import matplotlib.pyplot as plt
from evc import eigenvector_continuation as evc
from evc import lec_values as lecs
from pyDOE import lhs
import SALib.sample.sobol as sobol_design
import SALib.sample.latin as latin_design
import SALib.sample.finite_diff as fd_design
import sobol_core as sobol_core
import json
import os
import prettyplease

np.random.seed(666)

parameter_names_W_const = list(np.copy(lecs.Delta_NNLOgo_394_parameters_names))
parameter_names_Wo_const = list(np.copy(lecs.Delta_NNLOgo_394_parameters_names))
#constant first for reading filed in correct order
parameter_names_W_const.insert(0,'const')

#add the OPE_strength
parameter_names_W_const.append('OPE_strength')
parameter_names_Wo_const.append('OPE_strength')

print(parameter_names_W_const)

LECvalues = lecs.Delta_NNLOgo_394_LECs
#insert nominal value 1.0 for the OPE strength
LECvalues['OPE_strength'] = 1.0

domain_dimension = 18
subspace_dimension = 68

#run_case = 'ne20_68pts_30_percent_domain_no_isobreak_N10E16_hw14'
#path = f'./../spcc_data/{run_case}/'
#file_base_H = 'hf_evc_68pts_30_percent_domain_no_isobreak_%s_ham_mtx_N10E16_hw14_Ne20.dat'
#file_norm = 'hf_evc_68pts_30_percent_domain_no_isobreak_norm_mtx_N10E16_hw14_Ne20.dat'

#run_case = 'ne32_68pts_20_percent_domain_no_isobreak_N10E16_hw14'
#path = f'./../spcc_data/{run_case}/'
#file_base_H = 'hf_evc_68pts_20_percent_domain_no_isobreak_%s_ham_mtx_N10E16_hw14_Ne32.dat'
#file_norm = 'hf_evc_68pts_20_percent_domain_no_isobreak_norm_mtx_N10E16_hw14_Ne32.dat'

run_case = 'mg24_68pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14'
path = './../spcc_data/mg24_68pts_20_percent_domain_no_isobreak_OPE_vary_N10E16_hw14/'
file_base_H = 'hf_evc_68pts_20_percent_domain_no_isobreak_%s_ham_mtx_N10E16_hw14_Mg24.dat'
file_norm = 'hf_evc_68pts_20_percent_domain_no_isobreak_norm_mtx_N10E16_hw14_Mg24.dat'

em_UHF = evc.read_emulator_block(name='UHF', domain_dimension=domain_dimension, subspace_dimension=subspace_dimension, parameter_list=parameter_names_W_const,
                                      path = path, block=1, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True)

em_0 = evc.read_emulator_block(name='em_0', domain_dimension=domain_dimension, subspace_dimension=subspace_dimension, parameter_list=parameter_names_W_const,
                                    path = path, block=2, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True)

em_2 = evc.read_emulator_block(name='em_2', domain_dimension=domain_dimension, subspace_dimension=subspace_dimension, parameter_list=parameter_names_W_const,
                                    path = path, block=3, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True)

em_4 = evc.read_emulator_block(name='em_4', domain_dimension=domain_dimension, subspace_dimension=subspace_dimension, parameter_list=parameter_names_W_const,
                                    path = path, block=4, file_base_H = file_base_H, file_norm = file_norm, remove_constant=True)

# ToDo: add 'drop-states' functionality to emulator class
# include the states to eliminate ('drop') from emulator
#these_states = []
#these_states = [item for item in range(0, 34)]
#print(f'drop_states = these_states = {these_states}')

# test-point to reproduce Weiguangs results using emulators
zero_point = []
for idx, par in enumerate(parameter_names_Wo_const):
    try:
        zero_point.append(LECvalues[par])
    except KeyError:
        continue

def compute_overlap(eigvec, norm, H, no0b):
    NN = eigvec.T@norm@eigvec
    HH = eigvec.T@H@eigvec
    overlap = HH/NN
    return overlap.real + no0b

# include the states to eliminate ('drop') from emulator
these_states = []
#these_states = [item for item in range(0, 25)]
print(f'drop_states = these_states = {these_states}')

# compute zero-point values (DeltaGO)
eigval_UHF, eigvec_UHF, spectrum_UHF, H_UHF, norm_UHF,no0b_UHF = em_UHF.evaluate(hamiltonian_type = 'non-symmetric', domain_point = zero_point, drop_states = these_states)
eigval_0, eigvec_0, spectrum_0, H_0, norm_0,no0b_0 = em_0.evaluate(hamiltonian_type = 'non-symmetric', domain_point = zero_point, drop_states = these_states)
eigval_2, eigvec_2, spectrum_2, H_2, norm_2,no0b_2 = em_2.evaluate(hamiltonian_type = 'non-symmetric', domain_point = zero_point, drop_states = these_states)
eigval_4, eigvec_4, spectrum_4, H_4, norm_4,no0b_4 = em_4.evaluate(hamiltonian_type = 'non-symmetric', domain_point = zero_point, drop_states = these_states)

# compute zero-point values (DeltaGO)
print(f'eigval_UHF = {eigval_UHF}\t , overlap = {compute_overlap(eigvec_UHF,norm_UHF,H_UHF,no0b_UHF)}')
print(f'eigval_0   = {eigval_0}  \t , overlap = {compute_overlap(eigvec_UHF,norm_0  ,H_0  ,no0b_0)}')
print(f'eigval_2   = {eigval_2}  \t , overlap = {compute_overlap(eigvec_UHF,norm_2  ,H_2  ,no0b_2)}')
print(f'eigval_3   = {eigval_4}  \t , overlap = {compute_overlap(eigvec_UHF,norm_4  ,H_4  ,no0b_4)}')   

#
#cross validate deformed emulators to ensure everything is working
#
#def read_list_of_points(filename, skip_first = True):    
#    points = []
#    with open(filename,'r') as f:
#        if skip_first:
#            next(f)
#        for line in f:
#            myarray = np.fromstring(line, dtype=float, sep=' ')
#            points.append(myarray)
#    return np.array(points)
#
## c1 c2 c3 c4 Ct1S0np Ct1S0nn Ct1S0pp Ct3S1 C1S0 C3P0 C1P1 C3P1 C3S1 CE1 C3P2 cD cE
## ne20
##xval_points=read_list_of_points('./../spcc_data/ne20_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/list_of_points_Delta_go_394_mass_20_hw_14_10percent_400_points.txt')
##cc_vals = np.loadtxt('./../spcc_data/ne20_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/ne20_HFPAV_400ptns_10percent.txt')
#
## ne34
#xval_points=read_list_of_points('./../spcc_data/ne32_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/list_of_points_Delta_go_394_mass_20_hw_14_10percent_400_points.txt')
#cc_vals = np.loadtxt('./../spcc_data/ne32_HFPAV_400pts_10_percent_domain_no_isobreak_N10E16_hw14/ne32_HFPAV_400ptns_10percent.txt')
#
#exact_UHF = []
#exact_0   = []
#exact_2   = []
#exact_4   = []
#
#evc_UHF = []
#evc_0 = []
#evc_2 = []
#evc_4 = []
#
#
#index_list = list(cc_vals[:,0])
#index_list = [int(x) for x in index_list]
#for idx, this_point in enumerate(xval_points):
#    try :
#        array_index = index_list.index(idx)
#    except ValueError:
#        print(f'{idx} not in cc-exact values')
#        continue
#    #print(idx)
#    idx, E_unpro, E_0, E_2, E_4 = cc_vals[array_index,:]
#    #print(f'{idx}\t{E_unpro}\t{E_0}\t{E_2}\t{E_4}')
#
#    eigval_UHF, eigvec_UHF, spectrum_UHF, H_UHF, norm_UHF,no0b_UHF = em_UHF.evaluate(hamiltonian_type = 'non-symmetric', domain_point = this_point, drop_states = these_states)
#    eigval_0, eigvec_0, spectrum_0, H_0, norm_0,no0b_0             = em_0.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = this_point, drop_states = these_states)
#    eigval_2, eigvec_2, spectrum_2, H_2, norm_2,no0b_2             = em_2.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = this_point, drop_states = these_states)
#    eigval_4, eigvec_4, spectrum_4, H_4, norm_4,no0b_4             = em_4.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = this_point, drop_states = these_states)
#
#    overlap_UHF = compute_overlap(eigvec_UHF,norm_UHF,H_UHF,no0b_UHF) 
#    overlap_0   = compute_overlap(eigvec_UHF,norm_0  ,H_0  ,no0b_0)   
#    overlap_2   = compute_overlap(eigvec_UHF,norm_2  ,H_2  ,no0b_2)   
#    overlap_4   = compute_overlap(eigvec_UHF,norm_4  ,H_4  ,no0b_4)   
#    
#    #print(f'{idx}\t{eigval_UHF[0]}\t{eigval_0[0]}\t{eigval_2[0]}\t{eigval_4[0]}')
#    #print(f'{idx}\t{overlap_UHF}\t{overlap_0}\t{overlap_2-overlap_0}\t{overlap_4-overlap_0}\t{(overlap_4-overlap_0)/(overlap_2-overlap_0)}')
#    
#    exact_UHF.append(E_unpro) ; evc_UHF.append(overlap_UHF)
#    exact_0.append(E_0) ; evc_0.append(overlap_0)
#    exact_2.append(E_2) ; evc_2.append(overlap_2)
#    exact_4.append(E_4) ; evc_4.append(overlap_4)
#  
##exact = np.array(exact_UHF)
##evc   = np.array(evc_UHF)
##
##exact = np.array(exact_0)
##evc   = np.array(evc_0)
##
##exact = np.array(exact_2)
##evc   = np.array(evc_2)
##
##exact = np.array(exact_4)
##evc   = np.array(evc_4)
##
##exact = np.array(exact_2)-np.array(exact_0)
##evc   = np.array(evc_2)-np.array(evc_0)
##
##exact = np.array(exact_4)-np.array(exact_0)
##evc   = np.array(evc_4)-np.array(evc_0)
##
##exact = (np.array(exact_4)-np.array(exact_0))/(np.array(exact_2)-np.array(exact_0))
##evc   = (np.array(evc_4)-np.array(evc_0))/(np.array(evc_2)-np.array(evc_0))
#
#print(evc.shape)
#bar   = np.array(exact_UHF)
#f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8, 8))
#c = ax.scatter(exact,evc,c=bar,alpha=0.6)
#cbar = plt.colorbar(c)
#cbar.set_label('Egs', rotation=270)
#xy = np.linspace(*ax.get_xlim())
#ax.plot(xy, xy, color='k',lw=1)
#ax.set_xlabel('Exact');
#ax.set_ylabel('Emulator');
#plt.title('Ratio: Excitation energy 4+/2+')
#plt.show();

#############################
# define a parameter domain #
#############################

#domain_type = 'percentage'
#domain_type = 'NIPb208_limits'
domain_type = 'NIO28_limits'
#domain_type = 'iid_normal'
scale_factor = 0.5
lecs_to_vary = ['all']
use_roy_steiner = True
# cE in NNLOsat/NNLOgo is too small. Increase range by - factor
cE_factor = 1000
cD_factor = 25

C1S0_factor = 1

# you can also scale the Ct1S0* and Ct3S1 LECs. this will decrease their ranges
# good for keeping them within a narrow window while enlarging the percent_domain
CtLO_factor = 0.25

mid_point, lim_lo, lim_hi = lecs.setup_parameter_domain(LECvalues, domain_type, use_roy_steiner, scale_factor, CtLO_factor, C1S0_factor, cE_factor, cD_factor, lecs_to_vary)

mid_point = list(mid_point)
lim_lo = list(lim_lo)
lim_hi = list(lim_hi)

mid_point.append(1.0)
lim_lo.append(0.9999999)
lim_hi.append(1.0)

mid_point = np.array(mid_point)
lim_lo = np.array(lim_lo)
lim_hi = np.array(lim_hi)

print(f'SAMPLING IN:')
print(f'domain_type = {domain_type}')
print(f'scale_factor = {scale_factor}')

#print(' Using Roy-Steiner limits for c_1,2,3,4 boundaries\n')
if domain_type == 'iid_normal':
    print('%10s %12s   [%12s <--> %12s]\n'%('LEC','mid-point','mean','std dev'))
else:
    print('%10s %12s   [%12s <--> %12s]\n'%('LEC','mid-point','lower-limit','upper-limit'))
for i in range(0,domain_dimension):
    print('%10s %12.6f   [%12f <--> %12f]'%(parameter_names_Wo_const[i], mid_point[i], lim_lo[i], lim_hi[i]))

    
##########################
# prepare sobol sampling #
##########################

from tqdm import tqdm

print(f'Sobol analsysis')

Nexp=10
Nsamples = 2**Nexp

bounds_list = []
for i in range(0,domain_dimension):
    bounds_list.append([lim_lo[i],lim_hi[i]])

grouped = False
if grouped:
    grouped_parameters = ['piN','piN','piN','piN','LO_IV','LO_IV','LO_IV','LO_IS','NLO_IV','NLO_IV','NLO_IS','NLO_IV','NLO_IS','NLO_IS','NLO_IV','NNN','NNN']

if domain_type == 'iid_normal':
    dist_type = ['norm']*domain_dimension
else:
    dist_type = ['unif']*domain_dimension
problem = {
    'num_vars': domain_dimension,
    'names': parameter_names_Wo_const,
    'bounds': bounds_list,
    'dists' : dist_type}#,
#    'groups': grouped_parameters}

# QMC sample design
print(f'generating {Nsamples} sample points')
design = sobol_design
options = {
    "calc_second_order":True,
    "scramble":True}
#design = latin_design
#options = {}
#design = fd_design
#options = {
#    "delta":0.001
#    }
print(design)
sample_points = design.sample(problem, Nsamples,**options)
#2*(domain_dimension+1)*Nsamples
Nsamples = len(sample_points)
print(f'Total sample size = {Nsamples}')

#
# plot sample design
#
#prettyplease_keywords = {'plot_type_2d':'scatter'}
#fig = prettyplease.corner(sample_points,**prettyplease_keywords)
#plt.show()

Y_R42_values = []
Y_E0_values = []
Y_E2_values = []
Y_E4_values = []
Y_Ex2_values = []
Y_Ex4_values = []

#analyze the samples and plot the results
parameters_label = [r'$c_1$',r'$c_2$',r'$c_3$',r'$c_4$',r'$\tilde{C}^{(np)}_{1S0}$',r'$\tilde{C}^{(nn)}_{1S0}$',r'$\tilde{C}^{(pp)}_{1S0}$',r'$\tilde{C}_{3S1}$',r'$C_{1S0}$',r'$C_{3P0}$',r'$C_{1P1}$',r'$C_{3P1}$',r'$C_{3S1}$',r'$C_{E1}$',r'$C_{3P2}$',r'$c_D$',r'$c_E$',r'$OPE$']
#parameters_label = [r'$\pi N$',r'LO_IV',r'LO_IS',r'NLO_IV',r'NLO_IS',r'NNN']
load_data = False 

if domain_type == 'DeltaGO_percentage':
    domain_type += f'_{percent_domain}'
if use_roy_steiner:
    domain_type += f'_RS'
if grouped:
    domain_type += '_grouped_pars'

file_name = f'sobol_data_{run_case}_Nexp_{str(Nexp)}_{domain_type}_scalefactor_{scale_factor}'

if load_data:
    print(f'loading {file_name}')
    file_data = json.load( open( file_name+".json" ) )
    problem = file_data['problem'] = problem
    NExp = file_data['Nexp'] = Nexp
    Nsamples = file_data['Nsamples'] = len(sample_points)
    parameters_label = file_data['parameters_label'] = parameters_label
    sample_points = file_data['sample_points']
    Y_E0_values  = file_data['Y_E0_values']  
    Y_E0_values  = file_data['Y_E2_values']  
    Y_E0_values  = file_data['Y_E4_values']  
    Y_R42_values = file_data['Y_R42_values'] 
    Y_Ex2_values = file_data['Y_Ex2_values'] 
    Y_Ex4_values = file_data['Y_Ex4_values']
    
else:
    exists = os.path.isfile('./'+file_name+'.json')
    if exists:
        print(f'<Error: file {file_name} already exists>')
        raise SystemExit(1)
    
    #run the sampling
    for idx, Xi in enumerate(tqdm(sample_points)):

        eigval_UHF, eigvec_UHF, spectrum_UHF, H_UHF, norm_UHF,no0b_UHF = em_UHF.evaluate(hamiltonian_type = 'non-symmetric', domain_point = Xi, drop_states = these_states)
        eigval_0, eigvec_0, spectrum_0, H_0, norm_0,no0b_0             = em_0.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = Xi, drop_states = these_states)
        eigval_2, eigvec_2, spectrum_2, H_2, norm_2,no0b_2             = em_2.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = Xi, drop_states = these_states)
        eigval_4, eigvec_4, spectrum_4, H_4, norm_4,no0b_4             = em_4.evaluate(  hamiltonian_type = 'non-symmetric', domain_point = Xi, drop_states = these_states)
        
        overlap_0   = compute_overlap(eigvec_UHF,norm_0  ,H_0  ,no0b_0)   
        overlap_2   = compute_overlap(eigvec_UHF,norm_2  ,H_2  ,no0b_2)   
        overlap_4   = compute_overlap(eigvec_UHF,norm_4  ,H_4  ,no0b_4)   
        
        Y_E0 = overlap_0
        Y_E2 = overlap_2
        Y_E4 = overlap_4
        Y_Ex2 = Y_E2-Y_E0
        Y_Ex4 = Y_E4-Y_E0
        
        Y_R42 = Y_Ex4/Y_Ex2
        
        Y_R42_values.append(Y_R42)
        Y_E0_values.append(Y_E0)
        Y_E2_values.append(Y_E2)
        Y_E4_values.append(Y_E4)
        Y_Ex2_values.append(Y_Ex2)
        Y_Ex4_values.append(Y_Ex4)

    file_data = {}
    
    file_data['problem'] = problem
    file_data['Nexp'] = Nexp
    file_data['Nsamples'] = len(sample_points)
    file_data['parameters_label'] = parameters_label
    file_data['sample_points'] = sample_points.tolist()
    file_data['Y_E0_values'] = Y_E0_values
    file_data['Y_E2_values'] = Y_E0_values
    file_data['Y_E4_values'] = Y_E0_values
    file_data['Y_R42_values'] = Y_R42_values
    file_data['Y_Ex2_values'] = Y_Ex2_values
    file_data['Y_Ex4_values'] = Y_Ex4_values
    
    print('saving :%s'%file_name)
    json.dump( file_data, open(file_name+".json", 'w' ) )
    print(f'done')

sample_points = np.array(sample_points)

#analyze the samples and print the results
#Si_mean, Si_ci, St_mean, St_ci = sobol_core.sensitivity_analysis(problem, Y_E0_values, parameter_names_Wo_const, print_correlations=False)
#Si_mean, Si_ci, St_mean, St_ci = sobol_core.sensitivity_analysis(problem, Y_Ex2_values, parameter_names_Wo_const, print_correlations=False)
#Si_mean, Si_ci, St_mean, St_ci = sobol_core.sensitivity_analysis(problem, Y_Ex4_values, parameter_names_Wo_const, print_correlations=False)
#Si_mean, Si_ci, St_mean, St_ci = sobol_core.sensitivity_analysis(problem, Y_R42_values, parameter_names_Wo_const, print_correlations=False)

cutmin = -np.inf
cutmax = +np.inf
Y_R42_values = np.array(Y_R42_values)
print(np.count_nonzero(Y_R42_values<cutmin))
print(np.count_nonzero(Y_R42_values>cutmax))
mean_Y = Y_R42_values[(Y_R42_values>cutmin) & (Y_R42_values<cutmax)].mean()
std_Y = Y_R42_values[(Y_R42_values>cutmin) & (Y_R42_values<cutmax)].std()
print(mean_Y,std_Y)
np.random.seed(12347653)
Y_R42_values[Y_R42_values<cutmin] = mean_Y#np.random.lognormal(mean_Y,std_Y)
Y_R42_values[Y_R42_values>cutmax] = mean_Y#np.random.lognormal(mean_Y,std_Y)
#
fig_sensitivity = sobol_core.sensitivity_analysis_plot(problem, np.array(sample_points), Y_E0_values, Y_Ex2_values, Y_Ex4_values, Y_R42_values,
                                                       xlist_label=parameters_label,hist_ranges=[[-350,50],[0.0,5.0],[0.0,6],[0.0,8.0]])
plt.tight_layout()
plt.savefig('hm_plot.pdf',bbox_inches = 'tight', pad_inches = 0)
plt.show()

#Y = np.copy(Y_R42_values)
##Y = np.copy(Y_E0_values)
#Y = np.ma.masked_array(Y , mask = ((Y < cutmin) | (Y > cutmax)))
#fig_main_effects = sobol_core.main_effect_plot(problem, sample_points, Y, labels=parameters_label, Nr=5, Nc=4)
#plt.tight_layout()
#plt.savefig('main_effect_plot.jpg',bbox_inches = 'tight', pad_inches = 0)
#plt.show()

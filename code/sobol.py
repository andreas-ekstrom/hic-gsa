import numpy as np
import matplotlib.pyplot as plt
from emulator_lib import eigenvector_continuation as evc
from emulator_lib import lec_values as lecs
from pyDOE import lhs
import SALib.sample.sobol as sobol_design
import SALib.sample.latin as latin_design
import SALib.sample.finite_diff as fd_design
from SALib import ProblemSpec
import sobol_core as sobol_core
import json
import os
import prettyplease
import sys

parameters =  list(np.copy(lecs.NNLOsat_parameter_names))
parameters_wo_const = list(np.copy(lecs.NNLOsat_parameter_names))
#constant first
parameters.insert(0,'const')
LECvalues = lecs.NNLOsat_LECs

path = './../cc_output/spcc64_o16_nnlosat_emax6_hw18/spcc_matrices/'
file_base_H = 'hbar_20percent64_%s_nnlosat_mass_16_N06E16_hw16_OSC.dat'
file_norm = 'norm_20percent64_cE_nnlosat_mass_16_N06E16_hw16_OSC.dat'
files_obs = ['eccentricity_spcc.dat']
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

#############################
# define a parameter domain #
#############################

cE_factor = 20
cD_factor = 1
C1S0_factor = 1
CtLO_factor = 1

factor_overrides = {
    'cE': cE_factor,
    'cD': cD_factor,
    'C1S0': C1S0_factor,
    'CtLO': CtLO_factor,  # applies to Ct1S0pp/nn/np and Ct3S1
}

mode = 'percentage'
scale_factor = 0.1

mid_point, lim_lo, lim_hi = lecs.setup_parameter_domain(
    LECvalues=LECvalues,
    mode=mode,
    scale_factor=scale_factor,
    factor_overrides=factor_overrides,
)

print(f'SAMPLING IN:')
print(f'mode = {mode}')
print(f'scale_factor = {scale_factor}')


print('%10s %12s   [%12s <--> %12s]\n'%('LEC','mid-point','lower-limit','upper-limit'))
for i in range(0,domain_dim):
    print('%10s %12.6f   [%12f <--> %12f]'%(parameters_wo_const[i], mid_point[i], lim_lo[i], lim_hi[i]))

##########################
# prepare sobol sampling #
##########################

from tqdm import tqdm

print(f'SOBOL analsysis')

bounds_list = []
for i in range(0,domain_dim):
    bounds_list.append([lim_lo[i],lim_hi[i]])
    
problem_definition = {
    'num_vars': domain_dim,
    'names': parameters_wo_const,
    'bounds': bounds_list,
    'dists' : ['unif']*domain_dim}

# Initialize ProblemSpec
problem = ProblemSpec(problem_definition)

# Quasi-MC sample design
Nexp=12
Nsamples = 2**Nexp #base samples (expanded internally via Saltelli algo)
print(f'generating {Nsamples} sample points')
problem.sample_sobol(Nsamples, calc_second_order=True,scramble=True)
# scramble improves statistical properties (important for estimating confidence intervals).

#2*(domain_dimension+1)*Nsamples
sample_points = problem.samples
Nsamples = len(sample_points)
print(f'Total sample size = {Nsamples}')

#
# plot sample design
#
#prettyplease_keywords = {'plot_type_2d':'scatter'}
#fig = prettyplease.corner(sample_points,**prettyplease_keywords)
#plt.show()

Y_energy_values = []
Y_eccentricity_values = []

#analyze the samples and plot the results
parameters_label = [r'$c_1$',r'$c_3$',r'$c_4$',r'$\tilde{C}^{(np)}_{1S0}$',r'$\tilde{C}^{(nn)}_{1S0}$',r'$\tilde{C}^{(pp)}_{1S0}$',r'$\tilde{C}_{3S1}$',r'$C_{1S0}$',r'$C_{3P0}$',r'$C_{1P1}$',r'$C_{3P1}$',r'$C_{3S1}$',r'$C_{E1}$',r'$C_{3P2}$',r'$c_D$',r'$c_E$']
load_data = True 

file_name = f'sobol_data_Nexp_{str(Nexp)}_{mode}_scalefactor_{scale_factor}'

if load_data:
    print(f'loading {file_name}')
    file_data = json.load( open( file_name+".json" ) )
    problem = file_data['problem'] = problem
    NExp = file_data['Nexp'] = Nexp
    Nsamples = file_data['Nsamples'] = len(sample_points)
    parameters_label = file_data['parameters_label'] = parameters_label
    sample_points = file_data['sample_points']
    Y_energy_values  = file_data['Y_energy_values']  
    Y_eccentricity_values  = file_data['Y_eccentricity_values']  
else:
    exists = os.path.isfile('./'+file_name+'.json')
    if exists:
        print(f'<Error: file {file_name} already exists>')
        raise SystemExit(1)
    
    #run the sampling
    for idx, Xi in enumerate(tqdm(sample_points)):

        obs, eigvec_O, spectrum = em.evaluate(Xi,level=0)


        Y_energy_values.append(spectrum[0].real)
        Y_eccentricity_values.append(obs[0])

    file_data = {}
    
    file_data['problem'] = problem
    file_data['Nexp'] = Nexp
    file_data['Nsamples'] = len(sample_points)
    file_data['parameters_label'] = parameters_label
    file_data['sample_points'] = sample_points.tolist()
    file_data['Y_energy_values'] = Y_energy_values
    file_data['Y_eccentricity_values'] = Y_eccentricity_values
 
    
    print('saving :%s'%file_name)
    json.dump( file_data, open(file_name+".json", 'w' ) )
    print(f'done')


#plt.scatter(Y_energy_values,Y_eccentricity_values,color='black',alpha=0.05)
#plt.xlabel('energy',fontsize=15)
#plt.ylabel('eccentricity',fontsize=15)
#plt.savefig('energy_vs_eccentricity.pdf')
#plt.show()
#
#sys.exit(-1)
    
#analyze the samples and print the results
S1_mean, S1_ci, ST_mean, ST_ci, analyzed = sobol_core.sensitivity_analysis(problem, Y_energy_values, print_correlations=False)
#S1_mean, S1_ci, ST_mean, ST_ci, analyzed = sobol_core.sensitivity_analysis(problem, Y_eccentricity_values, print_correlations=False)

parameters_label = [r'$c_1$',r'$c_3$',r'$c_4$',r'$\tilde{C}^{(np)}_{1S0}$',r'$\tilde{C}^{(nn)}_{1S0}$',r'$\tilde{C}^{(pp)}_{1S0}$',r'$\tilde{C}_{3S1}$',r'$C_{1S0}$',r'$C_{3P0}$',r'$C_{1P1}$',r'$C_{3P1}$',r'$C_{3S1}$',r'$C_{E1}$',r'$C_{3P2}$',r'$c_D$',r'$c_E$']

fig_heatmap = sobol_core.sensitivity_analysis_plot_second_order_heatmap(problem=problem,               # your ProblemSpec with samples set
                                                                        Y_values=Y_eccentricity_values,      # the output you want analyzed/visualized
                                                                        xlist_label=parameters_wo_const,
                                                                        thresh=0.001,
                                                                        triangle="upper",
                                                                        annotate=False,
)

fig_heatmap.tight_layout()
fig_heatmap.savefig('heatmap_plot.pdf',bbox_inches = 'tight', pad_inches = 0)
plt.show()



fig_sensitivity = sobol_core.sensitivity_analysis_plot_multi(problem,
                                                             Y_values_list = [Y_energy_values, Y_eccentricity_values],    # list of length Ny, each an array-like of model outputs
                                                             xlist_label = parameters_label,                              # list of parameter labels (LaTeX ok)
                                                             hist_ranges=None,                                            # list of (lo, hi) ranges per Y; or None for auto
                                                             y_labels= [r'$E$',r'$\epsilon$'],                            # list of legend / histogram xlabels (e.g., [r'$E(0^+)$', ...])
                                                             width=None,                                                  # bar width; if None, chosen based on Ny
                                                             capsize=2,                                                   # error bar cap size
                                                             )

fig_sensitivity.tight_layout()
plt.savefig('hm_plot.pdf',bbox_inches = 'tight', pad_inches = 0)
plt.show()


from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import os as os
import LECvalues as LECvalues

def lhs_points(dim,npts):
    c_vec = lhs(dim, npts,criterion='maximin')
    return c_vec

def adjust(x, lo, hi):
  return np.array(x) * (hi - lo) + lo

np.random.seed(666)

#############################################################################################
### INPUT STARTS HERE
#############################################################################################

# select the number of sample points
npts = 64

percent_domain = 10

# cc input
twobody_nmax = 6 
hbar_omega = 18  
occ_protons = 3  
occ_neutrons = 3 
mass_nucleus = 16

# cE in NNLOsat/NNLOgo is too small. Increase range by x50 - factor
cE_factor = 200
cD_factor = 10
# you can also scale the Ct1S0* and Ct3S1 LECs. this will decrease their ranges
# good for keeping them within a narrow window while enlarging the percent_domain
CtLO_factor = 0.25

# isospin breaking switch
no_isobreak = True
# is True, tune the CIB (nn-pp) and CSB (nn-pp-np) differences
# 0.005 corresponds to allowing for 0.5% CSB/CIB breaking in the LECs
# 0.5% is a reasonable value (comparable to realistic chiral interactions)
CIBlim = 0.005
CSBlim = 0.005

#############################################################################################
### INPUT ENDS HERE
#############################################################################################

#add the domain percentage in directories and files                                                                                                
name = '%dpercent'%percent_domain

#_name_pnt_nofpnts_pnts                                                                                                                                    
suffix = '_%s_%d_%d_pnts.ini'

# prefix all files and directories
name_prefix = 'NNLOsat_mass_%d_hw_%d'%(mass_nucleus,hbar_omega)

lecs_to_vary = ['all']

# sampling selected LECs not yet working with no_isobreak = True 
#lecs_to_vary = ['C1S0']


# if you select another set of LECs this will
# require you to change also lambda and the structure
# of the CC input-files below

LECvalues = LECvalues.NNLOsat_LECs

parameter_names = list(LECvalues.keys())
dim_full_domain = len(LECvalues)
zero_point = np.ones(dim_full_domain)
for idx, par in enumerate(parameter_names):
    zero_point[idx] = LECvalues[par]

lim_lo = []
lim_hi = []

dim = len(LECvalues)
domain_size_factor = percent_domain/100.0

header = ''

if no_isobreak:
    # since we scale the NNLOsat values, construct a symmetric LO-1S0 coupling
    # percent_domain*Ctsym*CtLO_factor  will set the limits for the LO-1S0 limits
    Ctsym = 0.5*(0.5*(LECvalues['Ct1S0pp'] + LECvalues['Ct1S0nn']) + LECvalues['Ct1S0np'])

    LECvalues['Ct1S0pp'] = Ctsym
    LECvalues['Ct1S0nn'] = Ctsym
    LECvalues['Ct1S0np'] = Ctsym
            
for lec, val in LECvalues.items():
    header += lec +' '

    domain_size = domain_size_factor
    if lec == 'cE':
        domain_size = domain_size_factor * cE_factor

    if lec == 'cD':
        domain_size = domain_size_factor * cD_factor

    if lec == 'Ct1S0pp':
        domain_size = domain_size_factor * CtLO_factor

    if lec == 'Ct1S0nn':
        domain_size = domain_size_factor * CtLO_factor

    if lec == 'Ct1S0np':
        domain_size = domain_size_factor * CtLO_factor

    if lec == 'Ct3S1':
        domain_size = domain_size_factor * CtLO_factor * 0.5

    if lecs_to_vary == ['all']:
        lim = domain_size * np.abs(val)
    else:
        lim = domain_size * np.abs(val)
        if lec not in lecs_to_vary:
            lim = 0.0

    lim_lo.append(val - lim)
    lim_hi.append(val + lim)

c1_idx      = list(LECvalues.keys()).index("c1")
#c2_idx      = list(LECvalues.keys()).index("c2")
c3_idx      = list(LECvalues.keys()).index("c3")
c4_idx      = list(LECvalues.keys()).index("c4")
Ct1S0nn_idx = list(LECvalues.keys()).index("Ct1S0nn")
Ct1S0np_idx = list(LECvalues.keys()).index("Ct1S0np")
Ct1S0pp_idx = list(LECvalues.keys()).index("Ct1S0pp")
Ct3S1_idx   = list(LECvalues.keys()).index("Ct3S1")
C1S0_idx    = list(LECvalues.keys()).index("C1S0")
C3P0_idx    = list(LECvalues.keys()).index("C3P0")
C1P1_idx    = list(LECvalues.keys()).index("C1P1")
C3P1_idx    = list(LECvalues.keys()).index("C3P1")
C3P2_idx    = list(LECvalues.keys()).index("C3P2")
C3S1_idx    = list(LECvalues.keys()).index("C3S1")
CE1_idx     = list(LECvalues.keys()).index("CE1")
cD_idx      = list(LECvalues.keys()).index("cD")
cE_idx      = list(LECvalues.keys()).index("cE")

lp = lhs_points(dim, npts)
for j in range(npts):
    for i in range(dim):
        lp[j, i] = adjust(lp[j, i], lim_lo[i], lim_hi[i])

if no_isobreak:

    def random_value(x0,tol):
        rnd = np.random.random()
        pnt = x0 - (1.0 - 2.0*rnd)*tol
        return pnt
    
    for j in range(npts):

        # start from Ct1S0nn
        Cnn = lp[j,Ct1S0nn_idx]
        # draw pp value such that |nn-pp|/nn < CSBlim
        Cpp = random_value(Cnn,CSBlim*np.abs(Cnn))
        # dram np value suc that |0.5*(nn+pp) - np|/|| < CIBlim
        Cnp = random_value(0.5*(Cnn+Cpp),CIBlim*np.abs(0.5*(Cnn+Cpp)))

        lp[j,Ct1S0pp_idx] = Cpp
        lp[j,Ct1S0np_idx] = Cnp
        
    name_prefix = name_prefix + '_NO_ISOBREAK'

print('\n SAMPLING FROM PARAMETER DOMAIN \n')
print('%10s %20s   [%12s <--> %12s]\n'%('LEC','zero-point','lower-limit','upper-limit'))
for i in range(0,dim_full_domain):
    print('%10s %20.16f   [%12f <--> %12f]'%(parameter_names[i], zero_point[i], lim_lo[i], lim_hi[i]))

if no_isobreak:
    print('\n NO ISOSPIN-BREAKING. nn-np-pp LECs drawn within %.2f%% of eachother'%(CSBlim*100))
    
#add the domain percentage in directories and files
domain_type = '%dpercent'%percent_domain

#_name_pnt_nofpnts_pnts
suffix = '_%s_%d_%d_pnts.ini'

print("\ngenerating %d LHS points\n"%npts)

# get pwd
path = os.getcwd()+'/'
#print ("The current working directory is %s" % path)

# create output directory where to print the ini files
output_dir = 'cc_lhs_input_'+name_prefix+'_%d_percent_%d_points'%(percent_domain,npts)

if lecs_to_vary != ['all']:
    for lec in lecs_to_vary:
        output_dir+='_'+lec

output_dir += '/'
    
os.mkdir(path+output_dir)

print("written to files in:",path+output_dir)

np.savetxt(path+output_dir+'list_of_points_'+name_prefix+'_%dpercent_%d_points.txt'%(percent_domain,npts),lp,header=header)
filenames = []
print()
for pnt in range(npts):
    filename = path+output_dir+'cc_lhs_input_'+name_prefix+suffix%(domain_type,pnt,npts)
    f = open(filename,'w+')
    #print(filename)
    filenames.append('cc_lhs_input_'+name_prefix+suffix%(domain_type,pnt,npts))
    f.write('# specify single-particle data and model-space parameters\n')
    f.write('twobody_nmax = %d\n'%twobody_nmax)
    f.write('hbar_omega = %d\n'%hbar_omega)
    f.write('occ_protons = %d\n'%occ_protons)
    f.write('occ_neutrons = %d\n'%occ_neutrons)
    f.write('mass_nucleus = %d\n'%mass_nucleus)
    f.write('sp_nocc_cut = 0.0\n')
    f.write('\n')
    f.write('# Input files:\n')
    f.write('hf_format = spcc\n')
    f.write('hf_input_file_orbits = sp_energy_nnlosat_N06E16_hw18_O16_OSC.dat\n')
    f.write('hf_input_file_onebody = kinetic_nnlosat_N06E16_hw18_O16_OSC.dat\n')
    f.write('hf_input_file_coefficients = umat_nnlosat_N06E16_hw18_O16_OSC.dat\n')

    f.write('\n')
    f.write('# Use pre calculated fock-matrix = yes/no\n')
    f.write('pre_calculated_fock_matrix = yes\n')
    f.write('\n')
    f.write('\n')
    f.write('# read in hf coeffs\n')
    f.write('read_hf_transformation_file = yes\n')
    f.write('\n')
    
    f.write('# number of iterations for ccm ground-state\n')
    f.write('ccm_iter = 1000\n')
    f.write('\n')
    f.write('# convergence criteria for ccm ground-state calculations\n')
    f.write('ccm_tolerance = 1.D-6\n')
    f.write('\n')
    f.write('# input parameters for diis algorithm to improve ccsd convergence \n')
    f.write('diis_subspace  = 10\n')
    f.write('diis_step_size = 10\n')
    f.write('diis_mixing = 0.5\n')
    f.write('\n')
    f.write('# type of cc ground-state calculation, default: ccsd\n')
    f.write('ccm_ground_state_calculation = ccsd\n')
    f.write('\n')
    f.write('# e3max cut for ground-state ccsdt and eom calculations\n')
    f.write('ccsdt_e3max_cut =  0\n')
    f.write('ccsdt_l3max_cut =  100\n')
    f.write('# file for t-amplitudes and l1 and l2 g.s. amplitudes\n')
    f.write('tl_file = t2_l2_ccsd_ccsd_n3loEM_srg18_N02_hw18_O16_OSC.h5 \n')

    f.write('# set whether ground-state amplitudes are precalculated\n')
    f.write('pre_calculated_groundstate = no\n')
    f.write('\n')
    f.write('# type of cc-eom calculations: lit/0vbb/eomccsd(t)/transitions/transitionspt2/pa_eomccsd(t)\n')
    f.write('ccm_eom_calculation = groundstate_linear\n')
    f.write('gs_density_file = Ca48_raddens_ccsd_nnlo394_delta_N06E14_hw16.dat\n')
    f.write('eom_approximation = ccsd\n')
    f.write('eom_vector_stored = no\n')
    f.write('eom_state_number = 1\n')
    f.write('eom_e3max_cut   = 0\n')
    f.write('eom_l3max_cut   = 100\n')
    f.write('eom_occnum3_cut = -1.0\n')
    f.write('eom_vec_file = He8_eom_vec_1plus.dat\n')
    f.write('bloch_horowitz_switch = no\n')
    f.write('\n')
    f.write('eom_occupation_numbers_stored = no\n')
    f.write('eom_occupation_numbers = no\n')
    f.write('eom_occupation_number_file = Sn100_eom_occupations_EM3_N06E16_hw16.dat\n')
    f.write('\n')
    f.write('# use normal ordered 1-b approximation: yes/no\n')
    f.write('normalordered_onebody_approx = yes\n')
    f.write('two_body_current_cD_fit = no\n')
    f.write('two_body_current_cD_val = 1.214\n')
    f.write('twobody_current_cD1_no1b_approx = mec_no1b_tz+1_Park2003nl0_HebelerLamreg394_cD1_minus_cD0_EM4_N06E16_hw16_Sn100.dat\n')
    f.write('twobody_current_no1b_approx = gt_no1b_tz+1_NNn4lo500-srg2.0_N06E14_hw16_O14.dat \n')
    f.write('\n')
    f.write('# speficy quantum numbers for equation-of-motion calculation and parameters for arnoldi algorithm\n')
    f.write('j_eom    =  0\n')
    f.write('ipar_eom =  0\n')
    f.write('tz_eom   =  0\n')
    f.write('number_of_states = 5\n')
    f.write('arnoldi_iter = 300\n')
    f.write('arnoldi_tolerance = 1.D-6\n')
    f.write('\n')
    f.write('# transition type: gamowteller, E2, dipole, etc...\n')
    f.write('transition = pertE\n')
    f.write('\n')
    f.write('# bare or similarity transformed operator\n')
    f.write('similarity_transformed_operator = yes\n')
    f.write('\n')
    f.write('# type of expectation value for eom-ccsd: rn/rp/rm/hcom/N*N\n')
    f.write('eom_expectation_value = rn\n')
    f.write('\n')
    f.write('# lecs for sp-cc\n')
    f.write('lec_file = lecs_mass_nucleus_%d_sc%d.txt\n'%(mass_nucleus,pnt))
    f.write('\n')
    f.write('# generate hdf5 no2b yes/no\n')
    f.write('generate_hdf5_no2b_split = no\n')
    f.write('eigenvector_continuation = no\n')

    f.close()

#write lec_files

prefix = 'lecs_mass_nucleus_%d_sc%d'
for pnt in range(npts):
    filename = path+output_dir+prefix%(mass_nucleus,pnt)+'.txt'
    f = open(filename,'w+')

    f.write('file_onebody_interaction1  = fock_nnlosat_c1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction2  = fock_nnlosat_c1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction3  = fock_nnlosat_c3_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction4  = fock_nnlosat_c4_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction5  = fock_nnlosat_Ct_3S1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction6  = fock_nnlosat_Ct_1S0pp_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction7  = fock_nnlosat_Ct_1S0np_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction8  = fock_nnlosat_Ct_1S0nn_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction9  = fock_nnlosat_C_1S0_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction10 = fock_nnlosat_C_3P0_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction11 = fock_nnlosat_C_1P1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction12 = fock_nnlosat_C_3P1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction13 = fock_nnlosat_C_3S1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction14 = fock_nnlosat_C_3S1-3D1_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction15 = fock_nnlosat_C_3P2_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction16 = fock_nnlosat_cD_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction17 = fock_nnlosat_cE_N06E16_hw18_O16_OSC.dat\n')
    f.write('file_onebody_interaction18 = fock_nnlosat_const_N06E16_hw18_O16_OSC.dat\n')
    f.write('\n')
    f.write('\n')
    f.write('file_twobody_interaction1  = vno2b_nnlosat_c1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction2  = vno2b_nnlosat_c1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction3  = vno2b_nnlosat_c3_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction4  = vno2b_nnlosat_c4_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction5  = vno2b_nnlosat_Ct_3S1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction6  = vno2b_nnlosat_Ct_1S0pp_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction7  = vno2b_nnlosat_Ct_1S0np_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction8  = vno2b_nnlosat_Ct_1S0nn_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction9  = vno2b_nnlosat_C_1S0_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction10 = vno2b_nnlosat_C_3P0_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction11 = vno2b_nnlosat_C_1P1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction12 = vno2b_nnlosat_C_3P1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction13 = vno2b_nnlosat_C_3S1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction14 = vno2b_nnlosat_C_3S1-3D1_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction15 = vno2b_nnlosat_C_3P2_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction16 = vno2b_nnlosat_cD_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction17 = vno2b_nnlosat_cE_N06E16_hw18_O16_OSC.h5\n')
    f.write('file_twobody_interaction18 = vno2b_nnlosat_const_N06E16_hw18_O16_OSC.h5\n')

    f.write('     \n')          
    f.write('cD =      %.16f\n'%lp[pnt,cD_idx])
    f.write('cE =      %.16f\n'%lp[pnt,cE_idx])
    f.write('c1 =      %.16f\n'%lp[pnt,c1_idx])
    f.write('c2 =      0.0 \n')
    f.write('c3 =      %.16f\n'%lp[pnt,c3_idx])
    f.write('c4 =      %.16f\n'%lp[pnt,c4_idx])
    f.write('LO_LEC1 =   %.16f\n'%lp[pnt,Ct3S1_idx])
    f.write('LO_LEC2 =   %.16f\n'%lp[pnt,Ct1S0pp_idx])
    f.write('LO_LEC3 =   %.16f\n'%lp[pnt,Ct1S0np_idx])
    f.write('LO_LEC4 =   %.16f\n'%lp[pnt,Ct1S0nn_idx])    
    f.write('NLO_LEC1 =  %.16f\n'%lp[pnt,C1S0_idx])
    f.write('NLO_LEC2 =  %.16f\n'%lp[pnt,C3P0_idx])
    f.write('NLO_LEC3 =  %.16f\n'%lp[pnt,C1P1_idx])
    f.write('NLO_LEC4 =  %.16f\n'%lp[pnt,C3P1_idx])
    f.write('NLO_LEC5 =  %.16f\n'%lp[pnt,C3S1_idx])
    f.write('NLO_LEC6 =  %.16f\n'%lp[pnt,CE1_idx])
    f.write('NLO_LEC7 =  %.16f\n'%lp[pnt,C3P2_idx])
    f.write('\n')
    f.write('spcc_radius_file = radius_const_spcc.dat\n')
    f.write('spcc_hbar_file = hbar_const_spcc.dat\n')
    f.write('spcc_norm_file = norm_spcc.dat\n')
    f.write('\n')
    f.write('read_uhf_file = no\n')
    f.write('no2b_hdf5_file = no2b_208Pb_nnlo394_deltago_split_HF_N10E22_hw12.h5\n')
    f.write('tl_file  = tl_ccsd_10percent_sc%d_%d_nnlosat_O16_N06E16_hw18.h5\n'%(pnt,npts))
    
    f.close()
#write batch job files

prefix = 'lhs_cc_%d.slurm'
counter=0
for idx in range(0,len(filenames),2):
    filename = path+output_dir+prefix%counter
    f = open(filename,'w+')
    #print(filename)

    f.write('#!/bin/bash\n')
    f.write('#SBATCH -A nph123\n')
    f.write('#SBATCH -N 2\n')
    f.write('#SBATCH -J evc_cc\n')
    f.write('#SBATCH -t 1:00:00\n')
    f.write('#SBATCH -o output%J\n')
    f.write('#SBATCH -e error%J\n')
    f.write('#SBATCH -p gpu\n')
    f.write('\n')
    f.write('export OMP_NUM_THREADS=4\n')
    f.write('work_dir=/lustre/orion/scratch/hagen/nph123/vint_runs\n')
    f.write('cd $work_dir\n')
    f.write('\n')
    f.write('date\n')
    f.write('\n')
    
    f.write('# for intel\n')
    f.write('module purge\n')
    f.write('module load DefApps\n')
    f.write('module load hdf5/1.10.6\n')
    f.write('module load intel\n')
    f.write('module load gsl\n')
    f.write('\n')
     
    fname1 = filenames[idx]
    f.write('srun -n4 -N1 ./prog_cc_jscheme.x %s > %s &\n'%(fname1,fname1.replace('.ini','.out')))
    f.write('')
    if idx+1<len(filenames):
        fname2 = filenames[idx+1]
        f.write('srun -n4 -N1 ./prog_cc_jscheme.x %s > %s &\n'%(fname2,fname2.replace('.ini','.out')))
    f.write('\n')
    f.write('wait')
    counter+=1
    f.close()

filename = path+output_dir+'submit_all.sh'
f = open(filename,'w+')
f.write('#!/bin/bash\n')
for idx in range(0,counter):
    filename = 'sbatch ' + prefix%idx
    f.write('%s\n'%filename)
f.close()

print('DONE')

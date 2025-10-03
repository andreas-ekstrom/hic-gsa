import numpy as np
import os as os
import LECvalues as LECvalues


#############################################################################################
### INPUT STARTS HERE
#############################################################################################

# cc input
twobody_nmax = 6
hbar_omega = 16
occ_protons = 3  
occ_neutrons = 3 
mass_nucleus = 16

#############################################################################################
### INPUT ENDS HERE
#############################################################################################

#add the domain percentage in directories and files                                                                                             
# prefix all files and directories
name_prefix = 'NNLOsat_mass_%d_hw_%d'%(mass_nucleus,hbar_omega)

LECvalues = LECvalues.NNLOsat_LECs
LECvalues['const'] = 1.0

parameter_names = list(LECvalues.keys())
dim_full_domain = len(parameter_names)

const_idx   = list(LECvalues.keys()).index("const")
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

unit_matrix = np.eye(dim_full_domain,dim_full_domain)
unit_matrix[const_idx,const_idx] = 0.0

print('\n SETTING UP SPLIT FILES \n')

# get pwd
path = os.getcwd()+'/'
#print ("The current working directory is %s" % path)

# create output directory where to print the ini files
output_dir = 'cc_split_input_'+name_prefix+'/'
    
os.mkdir(path+output_dir)

print("written to files in:",path+output_dir)

dir_and_filenames = []
filenames = []
print()
for idx, par in enumerate(parameter_names):
    dir_and_filename = path+output_dir+'split_input_%s.ini'%(par)
    filename = 'split_input_%s.ini'%(par)

    f = open(dir_and_filename,'w+')
    dir_and_filenames.append(dir_and_filename)
    filenames.append(filename)
        
    f.write('# specify single-particle data and model-space parameters\n')
    f.write('twobody_nmax = %d\n'%twobody_nmax)
    f.write('hbar_omega = %d\n'%hbar_omega)
    f.write('occ_protons = %d\n'%occ_protons)
    f.write('occ_neutrons = %d\n'%occ_neutrons)
    f.write('mass_nucleus = %d\n'%mass_nucleus)
    f.write('sp_nocc_cut = 0.0\n')
    f.write('\n')
    # Input files:
    f.write('# Input files:\n')
    f.write('hf_format = spcc\n')
    f.write('hf_input_file_orbits = sp_energy_nnlosat_N06E16_hw18_O16_OSC.dat\n')
    f.write('hf_input_file_onebody = kinetic_nnlosat_N06E16_hw18_O16_OSC.dat\n')
    f.write('hf_input_file_coefficients = umat_nnlosat_N06E16_hw18_O16_OSC.dat\n')

    f.write('\n')
    f.write('# Use pre calculated fock-matrix = yes/no\n')
    f.write('pre_calculated_fock_matrix = yes\n')
    f.write('\n')
    f.write('# read in hf coeffs\n')
    f.write('read_hf_transformation_file = yes\n')
    f.write('\n')
    
    f.write('# number of iterations for ccm ground-state\n')
    f.write('ccm_iter = 100\n')
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
    f.write('pre_calculated_groundstate = yes\n')
    f.write('\n')
    f.write('# type of cc-eom calculations: lit/0vbb/eomccsd(t)/transitions/transitionspt2/pa_eomccsd(t)\n')
    f.write('ccm_eom_calculation = groundstate\n')
    f.write('gs_density_file = Ca48_raddens_ccsd_nnlo394_delta_N06E14_hw16.dat\n')
    f.write('eom_approximation = ccsd\n')
    f.write('eom_vector_stored = no\n')
    f.write('inpeom_state_number = 1\n')
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
    f.write('number_of_states = 68\n')
    f.write('arnoldi_iter = 100\n')
    f.write('arnoldi_tolerance = 1.D-6\n')
    f.write('\n')
    f.write('# transition type: gamowteller, E2, dipole, etc...\n')
    f.write('transition = pertE\n')
    f.write('\n')
    f.write('# bare or similarity transformed operator\n')
    f.write('similarity_transformed_operator = yes\n')
    f.write('\n')
    f.write('# type of expectation value for eom-ccsd: rn/rp/rm/hcom/N*N\n')
    f.write('eom_expectation_value = eccentricity\n')
    f.write('\n')
    f.write('# lecs for sp-cc\n')
    f.write('lec_file = lecs_mass_nucleus_%d_%s.txt\n'%(mass_nucleus,par))
    f.write('\n')
    f.write('# generate hdf5 no2b yes/no\n')
    f.write('generate_hdf5_no2b_split = no\n')
    f.write('eigenvector_continuation = yes\n')

    f.close()

#write lec_files

prefix = 'lecs_mass_nucleus_%d_%s'
for idx,par in enumerate(parameter_names):
    filename = path+output_dir+prefix%(mass_nucleus,par)+'.txt'
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
    f.write('cD =      %.16f\n'%unit_matrix[cD_idx,idx])
    f.write('cE =      %.16f\n'%unit_matrix[cE_idx,idx])
    f.write('c1 =      %.16f\n'%unit_matrix[c1_idx,idx])
    f.write('c2 =      0.0 \n') #%.16f\n'%unit_matrix[c2_idx,idx])
    f.write('c3 =      %.16f\n'%unit_matrix[c3_idx,idx])
    f.write('c4 =      %.16f\n'%unit_matrix[c4_idx,idx])
    f.write('LO_LEC1 =   %.16f\n'%unit_matrix[Ct3S1_idx,idx])
    f.write('LO_LEC2 =   %.16f\n'%unit_matrix[Ct1S0pp_idx,idx])
    f.write('LO_LEC3 =   %.16f\n'%unit_matrix[Ct1S0np_idx,idx])
    f.write('LO_LEC4 =   %.16f\n'%unit_matrix[Ct1S0nn_idx,idx])  
    f.write('NLO_LEC1 =  %.16f\n'%unit_matrix[C1S0_idx,idx])
    f.write('NLO_LEC2 =  %.16f\n'%unit_matrix[C3P0_idx,idx])
    f.write('NLO_LEC3 =  %.16f\n'%unit_matrix[C1P1_idx,idx])
    f.write('NLO_LEC4 =  %.16f\n'%unit_matrix[C3P1_idx,idx])
    f.write('NLO_LEC5 =  %.16f\n'%unit_matrix[C3S1_idx,idx])
    f.write('NLO_LEC6 =  %.16f\n'%unit_matrix[CE1_idx,idx])
    f.write('NLO_LEC7 =  %.16f\n'%unit_matrix[C3P2_idx,idx])
 
    f.write('\n')
    f.write('spcc_radius_file = eccentricity_20percent64_%s_nnlosat_mass_%d_N06E16_hw%d_OSC.dat\n'%(par,mass_nucleus,hbar_omega))
    f.write('spcc_hbar_file = hbar_20percent64_%s_nnlosat_mass_%d_N06E16_hw%d_OSC.dat\n'%(par,mass_nucleus,hbar_omega))
    f.write('spcc_norm_file = norm_20percent64_%s_nnlosat_mass_%d_N06E16_hw%d_OSC.dat\n'%(par,mass_nucleus,hbar_omega))
    
    f.write('\n')
    f.write('number_spcc_vecs = 64\n')
    f.write('no2b_hdf5_file = no2b_208Pb_nnlo394_deltago_split_N10E22_hw%d.h5\n'%(hbar_omega))
    f.write('tl_file  = tl_ccsd.h5    \n')
    f.close()
#write batch job files

prefix = 'lhs_cc_split_%d.slurm'
counter=0
for idx in range(0,len(dir_and_filenames),2):
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
    #f.write('work_dir=/gpfs/alpine/nph123/scratch/hagen/CC_runs/evc-cc\n')
    #f.write('cd $work_dir\n')
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

filename = path+output_dir+'submit_split_all.sh'
f = open(filename,'w+')
f.write('#!/bin/bash\n')
for idx in range(0,counter):
    filename = 'sbatch ' + prefix%idx
    f.write('%s\n'%filename)
f.close()

print('DONE')

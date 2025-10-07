import numpy as np
import sympy as sp
import scipy
import pyDOE as pyDOE
import scipy.linalg as spla
from scipy.sparse.linalg import eigsh
import os
from datetime import datetime
import random

def scale_point(x, lo, hi):
    return np.array(x) * (hi - lo) + lo

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")     

#def drop_state_from_matrix(A, i):
#    return np.delete(np.delete(A, i, axis = 0), i, axis = 1)

def find_nearest(array, value):
    array = np.asarray(array)
    diff_arrays = np.abs(array - value)
    idx = (diff_arrays.argmin())
    idxs = np.where(diff_arrays == diff_arrays.min())[0]
    return idx, array[idx], idxs
        
class emulator:

    def __init__(self, name, domain_dimension, subspace_dimension):
        assert(name), '<Error: empty emulator name>'
        assert(domain_dimension>0),'<Error: negative domain dimension>'
        assert(subspace_dimension>0),'<Error: negative subspace dimension>'

        self.name = name
        self.domain_dimension = domain_dimension
        self.subspace_dimension = subspace_dimension

        self.parameter_names = []
        
        self.subspace_matrices = []
        self.subspace_observable = []
        self.subspace_norm_matrix = None

        self.training_points = None
        self.xval_points = None
                
        self.largespace_matrices = []
        self.largespace_dimension = None
        self.largespace_eigvecs = []
        self.largespace_eigvals = []
        self.largespace_observable = []
        
    def __str__(self):
        """ string representation of object """
        return '{0}: {1}, domain dimension: {2}'.format(self.__class__.__name__, self.name,self.domain_dimension)
    
    def __repr__(self):
        return self.__class__.__name__
        
    def set_parameter_names(self, parameter_names):
        self.parameter_names = parameter_names

    def set_training_domain(self, lim_lo, lim_hi):
        self.lim_train_lo = lim_lo
        self.lim_train_hi = lim_hi

    def set_xval_domain(self, lim_lo, lim_hi):
        self.lim_xval_lo = lim_lo
        self.lim_xval_hi = lim_hi
        
    def add_matrix(self, list_of_matrices, matrix, name, in_domain):
        assert(name), '<Error: empty matrix name>'
        assert(type(matrix) is np.ndarray), '<Error: matrix is not a numpy object>'
        assert(matrix.ndim == 2),'<Error: matrix is not 2D>'
        
        list_of_matrices.append([name, in_domain, matrix])
        print('appended %s matrix %s in_domain = %s'%(matrix.shape,name,bool(in_domain)))

    def add_largespace_matrix(self, matrix, name, in_domain=True):
        self.add_matrix(self.largespace_matrices, matrix, name, in_domain)
        self.largespace_dimension = matrix.shape

    def add_largespace_observable(self, matrix, name):
        self.add_matrix(self.largespace_observable, matrix, name, in_domain=True)
        
    def add_subspace_observable(self, matrix, name):
        self.add_matrix(self.subspace_observable, matrix, name, in_domain=True)
        
    def add_subspace_matrix(self, matrix, name, in_domain=True, remove_constant_term=False):
        if in_domain and remove_constant_term:
            assert(self.subspace_matrices[0][1] == False), 'Error: subspace-matrix 0 is in Domain (i.e. non-constant)'
            print('removing constant matrix')
            matrix -= self.subspace_matrices[0][2]
                
        self.add_matrix(self.subspace_matrices, matrix, name, in_domain)

    def add_subspace_norm_matrix(self, matrix):
        assert(type(matrix) is np.ndarray), '<Error: matrix is not a numpy object>'
        assert(matrix.ndim == 2),'<Error: matrix is not 2D>'
        det_matrix = np.linalg.det(matrix)
        print('-> norm matrix determinant  = %.16e' % det_matrix)
        print('-> norm matrix rank         = %d  ' % np.linalg.matrix_rank(matrix))
        if (det_matrix == 0.0):
            print('WARNING: zero determinant matrix')
            
        self.subspace_norm_matrix = matrix            

    def sum_parameter_times_matrix(self, list_of_matrices, parameter_values):
        # create empty matrix with same shape as the members in the list_of_matrices
        mtx = np.zeros(list_of_matrices[0][2].shape, dtype=float)

        offset = 0
        for idx, term in enumerate(list_of_matrices):
            if term[1] == False:
                value = 1
                offset += 1
            else:
                value = parameter_values[idx-offset]
                
            mtx += value * term[2]
            #print('%f x matrix-%s'%(value, term[0]))
            
        return mtx

    def drop_states_from_matrix(self, A, drop_states):
        """
        Return matrix with dropped states.

        Parameters
        ----------
        A : ndarray
            Full matrix.

        drop_states : int or array_like
            Single or iterable list of indices to remove from A.

        Returns
        -------
        ndarray
            Matrix with specified rows and columns removed.
            
        """
        return np.delete(np.delete(A, drop_states, axis = 0), drop_states, axis = 1)

    
    def expectation_value(self, bra, op, ket, N):

        if N is not None:
            norm = bra.transpose()@N@ket
        else:
            norm = bra.transpose()@ket
        return (bra.transpose()@op@ket)/norm

    
    def sample_select(self, domain_point):
    
        nof_states = self.subspace_dimension
        all_states = []
        all_states.extend(range(0,nof_states))
        
        nof_drops = 10
        random_drops = random.sample(all_states,nof_drops)
        spectrumB = self.evaluate(domain_point,drop_states=random_drops)[2]
        spectrumB = spectrumB[spectrumB.imag == 0]
        
        all_values = []
        for i in range(0,100):
            nof_drops = np.random.randint(5,10)
            random_drops = random.sample(all_states,nof_drops)
            spectrumA = self.evaluate(domain_point,drop_states=random_drops)[2]
            spectrumA = spectrumA[spectrumA.imag == 0]
            res = {i for i in spectrumA if np.isclose(spectrumB, i, atol=5.0).any()}
            all_values.extend(res)
            
        hist,hist_data = np.histogram(all_values, bins=100)
            
        return hist_data[hist.argmax()]
    
    def evaluate(self, domain_point, target=None, level=0, drop_states=None):
        """
        Evaluate emulator at a single domain point

        Returns:
            observables (array of floats): eigvals[0], obs_vals
        """

        sum_mtx = self.sum_parameter_times_matrix(self.subspace_matrices, domain_point)
        self.subspace_H_matrix = sum_mtx
        norm_mtx = self.subspace_norm_matrix

        if drop_states is not None:

            sum_mtx = self.drop_states_from_matrix(sum_mtx,drop_states)
            norm_mtx = self.drop_states_from_matrix(norm_mtx,drop_states)
            
        # solve generalized eigenvalue problem

        eigvals, eigvec_L, eigvec_R = spla.eig(sum_mtx,norm_mtx, left=True, right=True)
        # sort wrt real part (and if tied: wrt imaginary part)
        s = np.argsort(eigvals)
        spectrum = eigvals[s]

        if target is not None:
            state, spcc_value, states  = find_nearest(spectrum, target)
            #if len(states) > 1:
            #    print('WARNING: more than one best state!!')

            level = state
            
        obs_vals = []

        if self.subspace_observable:

            for subspace_obs in self.subspace_observable:
                obs_mtx = subspace_obs[2]

                if drop_states is not None:
                    obs_mtx = self.drop_states_from_matrix(obs_mtx,drop_states)

                obs_val = self.expectation_value(eigvec_R[:,s[level]], obs_mtx, eigvec_R[:,s[level]], norm_mtx)
                obs_vals.append(obs_val)

        observables = []
        for obs in obs_vals:
            if obs.imag == 0:
                observables.append(obs.real)
            else:
                observables.append(obs)
                        
        if eigvals[s[level]].imag == 0:
            energy_eigenvalue = np.real(eigvals[s[level]].real)
        else:
            energy_eigenvalue = eigvals[s[level]]

        return observables, eigvec_R[:,s[level]], spectrum
    
    def batch_evaluate(self, domain_points, progress=False, use_best_sample=False, select_level=0, drop_states=None):
        """
        Evaluate emulator for a (two-dimensional) array of LECs.
        
        Args:
            domain_points: Two-dimensional array. The 0-axis is different LEC arrays. The 1-axis is the number of LECS.

        Returns:
            observables (two-dimensional array of floats): eigvals[0], obs_vals
                Axis 0 is of length len(array_of_lecs)
                Axis 1 is of length len(observables), where observables =  eigvals[0], obs_vals
        """

        assert len(domain_points.shape)==2, "array_of_lecs must have two dimensions."


        observables = []
        nof_points = len(domain_points)

        for idx, domain_point in enumerate(domain_points):
            if progress:
                if (idx%2500 == 0):
                    print('sample %d of %d'%(idx,nof_points))
            if use_best_sample:
                stable_sample = self.sample_select(domain_point)
                output = self.evaluate(domain_point, target=stable_sample, level=select_level, drop_states=drop_states)[0]
            else:
                output = self.evaluate(domain_point, level=select_level, drop_states=drop_states)[0]
            if output[0].imag == 0.0:
                observables.append(output)
            else:
                pass
                #print('warning: discard imaginary ground-state energy')
                #observables.append(output)
                
        return np.array(observables)

    @staticmethod
    def scale_point(x, lo, hi):
        return np.array(x) * (hi - lo) + lo
    
    def generate_training_points(self):
        self.training_points = pyDOE.lhs(self.domain_dimension, self.subspace_dimension)
        for j in range(self.subspace_dimension):
            for i in range(self.domain_dimension):
                self.training_points[j, i] = self.scale_point(self.training_points[j, i], \
                    self.lim_train_lo[i], self.lim_train_hi[i])

    def generate_xval_points(self, Nxval):
        self.xval_points = pyDOE.lhs(self.domain_dimension, Nxval)
        for j in range(Nxval):
            for i in range(self.domain_dimension):
                self.xval_points[j, i] = self.scale_point(self.xval_points[j, i], \
                    self.lim_xval_lo[i], self.lim_xval_hi[i])

    # only hermitian matrices
    def solve_exact(self,point,lanczos=True):

        # construct large space matrix
        sum_mtx = self.sum_parameter_times_matrix(self.largespace_matrices, point)

        # diagonalize and store lowest eigenvalue with corresponding eigenvector
        if lanczos:
            eigvals, eigvecs = eigsh(sum_mtx,k=1,which='SA')
        else:
            eigvals, eigvecs = spla.eig(sum_mtx)
            
        s = np.argsort(eigvals)

        return eigvals[s[0]], eigvecs[:,s[0]]
    
    #so far: only hermitean matrices for training
    def train(self, lanczos=True):
        print('Training: %s, %d training points, %s large-space problem'\
                  %(self.name,self.subspace_dimension,self.largespace_dimension))
        for idx, point in enumerate(self.training_points):
            print('point %d of %d'%(idx,len(self.training_points)))
            eigval, eigvec = self.solve_exact(point, lanczos=True)
            
            self.largespace_eigvecs.append(eigvec)
            self.largespace_eigvals.append(eigval)

        self.largespace_eigvecs = np.column_stack(self.largespace_eigvecs)
        # for orthogonalizing the subspace basis straight away
        #self.largespace_eigvecs = spla.orth(self.largespace_eigvecs)
        # construct norm matrix
        self.subspace_norm_matrix = np.transpose(self.largespace_eigvecs).dot(self.largespace_eigvecs)
        # project large space matrices to eigenvector subspace
        for large_matrix in self.largespace_matrices:
            print('Projecting %s into %s subspace'%(large_matrix[0],self.subspace_norm_matrix.shape))
            self.subspace_matrices.append([ large_matrix[0], large_matrix[1],\
                np.transpose(self.largespace_eigvecs).dot(large_matrix[2].dot(self.largespace_eigvecs))])

        #if self.largespace_observable is not None:
        for largespace_obs in self.largespace_observable:
            obs_mtx = largespace_obs
            self.add_subspace_observable(np.transpose(\
                self.largespace_eigvecs).dot(obs_mtx[2].dot(\
                self.largespace_eigvecs)),obs_mtx[0])

    def xval(self):

        exact_E    = []
        emulator_E = []

        exact_O_vals    = []
        emulator_O_vals = []
        
        # loop over xval points
        for idx, xval_point in enumerate(self.xval_points):
            print('cross valdation point no, %d'%idx)
            emulator_values = self.evaluate(xval_point)
            emulator_E.append(emulator_values[0])
            exact_E_value, exact_evec = self.solve_exact(xval_point)
            exact_E.append(exact_E_value)
            if len(emulator_values)>1:
                emulator_O_vals.append(emulator_values[1:])
                exact_O_values = []
                for largespace_obs in self.largespace_observable:
                    exact_O_value = self.expectation_value(np.transpose(exact_evec), largespace_obs[2],exact_evec,None)
                    exact_O_values.append(exact_O_value)
                exact_O_vals.append(exact_O_values)
        return np.array(exact_E), np.array(emulator_E), np.array(exact_O_vals), np.array(emulator_O_vals)
            
    def write_emulator_to_file(self,meta_data):
        path = os.getcwd()
        print ("The current working directory is %s" % path)
        emulator_path = path+'/evc_input/'
        try:
            # Create target Directory
            os.mkdir(emulator_path)
            print("Directory " , emulator_path ,  " Created ") 
        except FileExistsError:
            print("Directory " , emulator_path ,  " already exists")

        this_emulator = emulator_path+self.name+'_Nsub'+str(self.subspace_dimension)
        already_exists = False
        try:
            # Create target Directory
            os.mkdir(this_emulator)
            print("Directory " , this_emulator,  " Created ") 
        except FileExistsError:
            print("Directory " , this_emulator ,  " already exists")
            already_exists = True
            
        if already_exists:
            answer = None
            while answer not in ("yes", "no"):
                answer = input("continue? [yes/no]: ")
                if answer == "yes":
                    pass
                elif answer == "no":
                    return './'
                else:
    	            print("Please enter yes or no.")
         
        for mtx in self.subspace_matrices:
            print('writing term:', mtx[0])
            np.savetxt(this_emulator+'/'+mtx[0],mtx[2],fmt='%.18e')
   
        print('writing norm matrix:', 'norm_matrix')
        np.savetxt(this_emulator+'/'+'norm_matrix',self.subspace_norm_matrix,fmt='%.18e')

        for subspace_obs in self.subspace_observable:
            print('writing observable matrix:', subspace_obs[0])
            np.savetxt(this_emulator+'/'+subspace_obs[0],subspace_obs[2],fmt='%.18e')

        # datetime object containing current date and time
        now = datetime.now()
 
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    
        with open(this_emulator+'/'+'emulator_data.txt', 'w') as f:
            f.write('EMULATOR DATA\n')
            f.write(dt_string)	
            f.write('\n name: %s\n'%(self.name))
            f.write(' domain dimension: %d\n'%(self.domain_dimension))
            f.write(' subspace dimension: %d\n'%(self.subspace_dimension))

            for item in meta_data:
                f.write("%s\n" % item)
        
        return this_emulator
                
def read_emulator(name, domain_dimension, subspace_dimension, parameter_list,
                  path, file_base_H, file_norm, remove_constant=False, files_obs=None, names_obs=None):
    
    m = emulator(name, domain_dimension=domain_dimension, subspace_dimension=subspace_dimension)

    m.set_parameter_names(parameter_list)

    for parameter in parameter_list:
        filename = path+file_base_H%parameter
        print('loading %s'%filename)
        mtx = np.loadtxt(filename)
        if parameter == 'const':
            m.add_subspace_matrix(mtx, name=parameter, in_domain=False, remove_constant_term=remove_constant)
        else:
            m.add_subspace_matrix(mtx, name=parameter, in_domain=True, remove_constant_term=remove_constant)
    
    filename = path+file_norm
    print('loading %s'%filename)
    mtx = np.loadtxt(filename)
    m.add_subspace_norm_matrix(mtx)

    if files_obs is not None:
        for idx, file_obs in enumerate(files_obs):
            mtx = np.loadtxt(path+file_obs)
            m.add_subspace_observable(mtx, name=names_obs[idx])

    return m

class SmallBatchVoting:

    def __init__(self, subspace_H_mtx, subspace_norm_mtx, subspace_spectrum, numbatches=100, batchsize=30, \
                     tol_real_relative=0.001, tol_real_relative_backup=0.01, \
                     tol_imag_relative=0.01, min_vote_fraction=0.2):
        """
        Class for performing small batch voting.

        Methods
        -------
        setup_small_batch_voting()
        run_small_batch_voting()
        count_votes()
        select_vote_winner()
        drop_states_from_matrix(A, drop_states)

        Attributes
        ----------
        subspace_H_mtx : ndarray, shape(subspace_dimension, subspace_dimension)
            Hamiltonian matrix projected on subspace
        subspace_norm_mtx : ndarray, shape(subspace_dimension, subspace_dimension)
            Norm matrix for subspace states
        subspace_spectrum : ndarray, shape(subspace_dimension)
            Subspace eigenspectra (possibly complex)
        subspace_dimension : int
            Subspace dimension.
        numbatches : int, default=100
            Number of small batches.
        batchsize : int, default=30
            Size of small batches. Must be smaller than subspace_dimension.
        tol_real_relative : float, default=0.001
            Relative tolerance for real part of a small batch energy to cast 
            one vote on a subspace state.
        tol_real_relative_backup : float, default=0.01
            Backup (max) relative tolerance for real part of a small batch energy to cast 
            one vote on a subspace state. Only used if the first test does not work.
        number_successful_runs : int
            Counts the number of successful runs (producing a winner).
        number_use_backup_tol : int
            Counts the number of successful uses of the backup, max real tolerance.
        tol_imag_relative : float, default=0.01
            Relative tolerance for imaginary part of subspace and small batch 
            energy to count as physical.
        min_vote_fraction : float, default=0.2
            Minimum fraction of the small batches that must have casted a vote
            on a state to allow it to be a winner.

        small_batches_dropstates : ndarray of int, shape(numbatches,subspace_dimension-batchsize)
            Subspace states to be dropped for each small batch.
        small_batches_spectra : ndarray of complex, shape(numbatches, batchsize)
            All small batches eigenspectra (sorted by real part)
        small_batches_votes : ndarray of int, shape(numbatches, subspace_dimension)
            Number of votes per eigenstate in the subspace spectrum.
        small_batches_votes_backup : ndarray of int, shape(numbatches, subspace_dimension)
            Number of votes per eigenstate in the subspace spectrum using the max tolerance.
        small_batches_vote_winner : tuple (float, int)
            Real part of the vote-vinning eigenvalue, Index of state
        small_batches_max_votes : int
            Max number of votes to winning or closest to winning state.
        """
 
        (nrows_H,ncols_H) = subspace_H_mtx.shape
        assert nrows_H==ncols_H, '<Error: subspace matrix must be square>'
        (nrows_N,ncols_N) = subspace_norm_mtx.shape
        assert nrows_N==ncols_N, '<Error: subspace matrix must be square>'
        assert nrows_N==nrows_H, '<Error: subspace H and norm matrices must be the same shape>'
        assert len(subspace_spectrum) == nrows_H, '<Error: The subspace spectrum must contain all eigenvalues>'

        for input in (numbatches, batchsize, tol_real_relative, tol_real_relative_backup, \
                          tol_imag_relative, min_vote_fraction):
            assert(input>0),'<Error: input must be a positive number>'
            
        self.subspace_H_mtx = subspace_H_mtx
        self.subspace_norm_mtx = subspace_norm_mtx
        self.subspace_spectrum = subspace_spectrum
        self.subspace_dimension = nrows_H

        self.numbatches = numbatches
        self.batchsize = batchsize
        self.tol_real_relative = tol_real_relative
        self.tol_real_relative_backup = tol_real_relative_backup
        self.tol_imag_relative = tol_imag_relative
        self.min_vote_fraction = min_vote_fraction

        self.rng = np.random.default_rng()

        # Setup the small batch voting procedure.
        self.setup_small_batch_voting()
        self.number_use_backup_tol = 0
        self.number_successful_runs = 0
        
    def __str__(self):
        """ string representation of object """
        return f'{self.__class__.__name__}: numbatches={self.numbatches}, batchsize={self.batchsize}'
    
    def __repr__(self):
        return self.__class__.__name__

    def run_and_vote(self):
        "Run and vote."
        assert self.small_batches_dropstates.shape == (self.numbatches,self.subspace_dimension-self.batchsize),\
            "The SmallBatchVoting instance must have been initialized before running."
        self.run_small_batch_voting()
        self.count_votes()
        self.select_vote_winner()
        

    def setup_small_batch_voting(self):
        """Setup small-batch voting."""
        assert scipy.special.comb(self.subspace_dimension,self.batchsize) \
          >= self.numbatches, "Cannot generate this many unique small batches."
        numdrops = self.subspace_dimension - self.batchsize
        dropstates = np.zeros((self.numbatches,numdrops), dtype=int)
        # Fill the dropstates array
        dropstates[0,:] = np.sort(self.rng.permutation(self.subspace_dimension)[:numdrops])
        for irow in range(1,self.numbatches):
            unique=False
            while unique==False:
                suggestion = np.sort(self.rng.permutation(self.subspace_dimension)[:numdrops])
                # check that this suggestion has not been picked before
                unique = 0 not in np.abs(dropstates[:irow,:]-suggestion).sum(axis=1)
            dropstates[irow,:] = suggestion
        self.small_batches_dropstates = dropstates # ndarray numbatches x (subspace_dimension - batchsize)
        
    def run_small_batch_voting(self):
        """
        Run small-batch voting.

        Attributes
        -------
        small_batches_spectra : ndarray, shape(numbatches, batchsize)
            All small batches eigenspectra (possibly complex)
        """
        self.small_batches_spectra = np.zeros((self.numbatches, self.batchsize), dtype=np.complex128) 
        for i_small_batch, drop_states in enumerate(self.small_batches_dropstates):
            small_batch_norm_mtx = self.drop_states_from_matrix(self.subspace_norm_mtx, drop_states)
            small_batch_H_mtx = self.drop_states_from_matrix(self.subspace_H_mtx, drop_states)
            # solve generalized eigenvalue problem
            eigvals, eigvec_L, eigvec_R = spla.eig(small_batch_H_mtx, \
                                            small_batch_norm_mtx, left=True, right=True)
            # sort wrt real part (and if tied: wrt imaginary part)
            s = np.argsort(eigvals)
            self.small_batches_spectra[i_small_batch,:] = eigvals[s]

    def count_votes(self):
        """
        Counts the votes of the small-batch voting.

        Attribute
        -------
        small_batches_votes : ndarray, shape(numbatches, self.subspace_dimension)
            Number of votes per eigenstate in the subspace spectrum
        small_batches_votes_backup : ndarray, shape(numbatches, self.subspace_dimension)
            Number of votes per eigenstate in the subspace spectrum with the backup (max) tolerance.
        """
        self.small_batches_votes = np.zeros(self.subspace_dimension, dtype=int)
        self.small_batches_votes_backup = np.zeros(self.subspace_dimension, dtype=int)
        # Only consider small batch states with small complex part
        small_imag = np.abs(self.small_batches_spectra.imag / self.small_batches_spectra.real) \
          < self.tol_imag_relative
        for i_subspace_state, energy in enumerate(self.subspace_spectrum):
            if np.abs(energy.imag / energy.real) > self.tol_imag_relative:
                continue
            # Find small batch states within relative real tolerance
            rel_diff = np.abs((self.small_batches_spectra.real-energy.real) / energy.real)
            close_real = rel_diff < self.tol_real_relative
            # Try also with the larger, backup tolerance
            close_real_backup = rel_diff < self.tol_real_relative_backup
            # Each small batch casts a vote if one small batch state has
            # (1) small imaginary part and (2) real part close to the considered state.
            # Note: max one vote for each subspace state from each small batch
            votes = np.any(np.logical_and(close_real, small_imag), axis=1)
            votes_backup = np.any(np.logical_and(close_real_backup, small_imag), axis=1)
            self.small_batches_votes[i_subspace_state] = votes.sum()
            self.small_batches_votes_backup[i_subspace_state] = votes_backup.sum()

    def select_vote_winner(self):
        """
        Select the winner of the small-batch voting.

        Attribute
        -------
        small_batches_vote_winner : tuple (float, int)
            Real part of the vote-vinning eigenvalue; np.nan if no winner. Index of wote winner; np.nan if no winner
        small_batches_max_votes : int
            Max number of votes to winning or closest to winning state.
        """
        # sort indices by number of votes (from max to min)
        sort_votes = np.argsort(-self.small_batches_votes)
        # check which ones are allowed to win
        ok_to_win = self.small_batches_votes / self.numbatches >= self.min_vote_fraction
        try:
            win_index = sort_votes[ok_to_win[sort_votes]][0]
            self.small_batches_vote_winner = (self.subspace_spectrum[win_index], win_index)
            self.small_batches_max_votes = self.small_batches_votes[win_index]
            self.number_successful_runs+=1
        except IndexError:
            # Try with the backup (max) tolerance
            sort_votes = np.argsort(-self.small_batches_votes_backup)
            ok_to_win = self.small_batches_votes_backup / self.numbatches >= self.min_vote_fraction
            try:
                win_index = sort_votes[ok_to_win[sort_votes]][0]
                self.small_batches_vote_winner = (self.subspace_spectrum[win_index], win_index)
                self.small_batches_max_votes = self.small_batches_votes_backup[win_index]
                self.number_use_backup_tol += 1
                self.number_successful_runs+=1
            except IndexError:
                self.small_batches_vote_winner = (np.nan, np.nan)
                self.small_batches_max_votes = self.small_batches_votes[sort_votes[0]]
            
    def drop_states_from_matrix(self, A, drop_states):
        """
        Return matrix with dropped states.

        Parameters
        ----------
        A : ndarray
            Full matrix.

        drop_states : int or array_like
            Single or iterable list of indices to remove from A.

        Returns
        -------
        ndarray
            Matrix with specified rows and columns removed.
            
        """
        return np.delete(np.delete(A, drop_states, axis = 0), drop_states, axis = 1)

    

# 

import numpy as np
#import cmath
#from cmath import exp as exp
from numpy import pi as PI
from numpy import exp as exp
from numpy import sin as sin
from numpy import cos as cos

import scipy.sparse as sp
from scipy.sparse import csc_matrix

import time
import sys


import parameters as param


class Hilbert_Space:
    
    """
    ----------------
    - Variables
    ----------------
    numlevels:          number of (internal) levels per site
    filling:            number of particles per site
    nstates:            number of states
    localhilbertsize:   
    hilbertsize:
    
    
    ----------------
    - Lists
    ----------------
    identities:         List of identities of different sizes
    states_list:        List of all states (in binary) of local Hilbert space
    binaries_list:      List of binary coding of states_list
    
    
    ----------------
    - Functions
    ----------------
    
    
    
    """
    
    def __init__(self,numlevels,filling,nsites):
        
        self.numlevels = numlevels
        self.filling = filling
        self.nsites = nsites
        
        self.define_states()
        self.compute_number_of_excitations (param.deg_g,param.deg_e)
        #self.store_identities()
        
        if param.output_stateslist: self.output_stateslist()
        
        """if param.output_stateslist and param.deg_i<=0:
            filename = '%s/liststates_fill%s_Ng%i_Ne%i.txt'%(param.outfolder, param.filling, param.deg_g, param.deg_e)
            with open(filename,'w') as f:
                f.write('# Row 1: state number | Row 2: occupation excited | Row 3: occupation ground.\n')
                f.write('# \n# \n')
                for ii in range(len(self.states_list)):
                    f.write('%i\n'%(ii))
                    np.savetxt(f,[list(self.states_list[ii][param.deg_g:])],fmt='%i')
                    np.savetxt(f,[list(self.states_list[ii][:param.deg_g])],fmt='%i')
                    f.write('\n')
                    
                    #output_data = [ [ tt ] + list(self.states_list[tt]) for tt in range(len(self.states_list)) ]
                    #np.savetxt(f,output_data,fmt='%i')
                    
            if self.nsites>1:
                filename = '%s/globalstates_nsites%i_fill%s_Ng%i_Ne%i.txt'%(param.outfolder, self.nsites, param.filling, param.deg_g, param.deg_e)
                with open(filename,'w') as f:
                    f.write('# Row 1: global state number |x> | Row 2: stat in computational basis |y1,y2,y3,..>\n')
                    f.write('# \n# \n')
                    for ii in range(self.hilbertsize):
                        f.write('%i\n'%(ii))
                        # Write hash in computational basis
                        y = ii
                        m = self.hilbertsize/self.localhilbertsize
                        index = int(y/m)
                        f.write('%i '%(index))
                        for jj in range(1,self.nsites):
                            y = y - index*m
                            m = m/self.localhilbertsize
                            index = int(y/m)
                            f.write('%i '%(index))
                        f.write('\n\n')"""
            
        
        
    def store_identities(self):
        """
        Stores a list with (sparse) identity matrix of different sizes for fast construction of sigma_matrix operators.
        """
        self.identities = []
        for ii in range(self.nsites):
            self.identities.append( sp.identity(self.localhilbertsize**ii) )
            
        
        
    def binary(self,state):
        """
        Expects array of 0s and 1s. Returns number represented by binary code given.
        """
        b = 0
        for ii in range(len(state)):
            b = b + state[ii]*2**ii
        return b
        
        
        
    def define_states (self):
        """
        Fermions
        
        Creates and sorts in increasing order of binary hash number
        - states_list: list of all states in the local Hilbert space in binary coding.
        - binaries_list: contains the binary equivalent of the states.
        
        Note: Each state is represented by a list of 0 and 1, where a 1 in entry n means that the single-body state |n> is singly occupied.
        The many-body Fock states |(b0,b1,...)^T> are defined as (f_0^dagger)^b0 (f_1^dagger)^b1 ... |vacuum>
        """
        
        self.states_list = [np.zeros(self.numlevels, dtype=int)]
        self.binaries_list = []
        temp = []
        
        # Create list of states for a given number of particles per site
        for nn in range(1,self.filling+1):
            temp=self.states_list
            self.states_list=[]
            for ii in range(len(temp)):
                
                # Find last entry with a 1
                if nn>1: rindex = np.argwhere(temp[ii] == 1).max()
                else: rindex = 0
        
                for jj in range(rindex,self.numlevels):
                    b = temp[ii] + 0
                    b[jj] = b[jj] + 1
                    if b[jj]<=1: self.states_list.append(b)
        
        # Compute binaries and indices of all states
        for ii in range(len(self.states_list)):
            self.binaries_list.append(self.binary(self.states_list[ii]))
        
        # Order states in increasing order of binaries
        ordering = sorted(range(len(self.binaries_list)),key=self.binaries_list.__getitem__)
        self.states_list = [self.states_list[ii] for ii in ordering]
        self.binaries_list = [self.binaries_list[ii] for ii in ordering]
        
        self.localhilbertsize = len(self.states_list)
        self.hilbertsize = self.localhilbertsize**self.nsites
        print("Local Hilbert size: %i"%(self.localhilbertsize))
        print("Hilbert size: %i"%(self.hilbertsize))
        print("Linblad size: %i"%(self.hilbertsize**2))
        
        #print(self.states_list)
        
    
    
    def compute_number_of_excitations (self,eindex,nestates):
        """
        Assuming that the internal levels are ordered as [ground levels, excited levels, other levels],
        compute and save for each state in states_list the number of excitations present between eindex and eindex+nestates.
        """
        self.excitations_list = []
        for ii in range(len(self.states_list)):
            self.excitations_list.append( sum(self.states_list[ii][eindex:eindex+nestates]) )
        
        
        
    def sigma_matrix (self,alpha,beta,ii):
        """
        Outputs (sparse) matrix representation of operator sigma_alphabeta^(ii) = f_alpha^dagger^(ii) f_beta^(ii) for current Hilbert space
        """
        
        return sp.kron( sp.kron( sp.identity(self.localhilbertsize**ii) , self.sigma_matrix_local( alpha , beta ) , format='csc'), sp.identity(self.localhilbertsize**(self.nsites-ii-1)) , format='csc')
        
        # Storing identities not much faster, apparently.
        #return sp.kron( sp.kron( self.identities[ii] , self.sigma_matrix_local( alpha , beta ) , format='csc'), self.identities[self.nsites-ii-1] , format='csc')
        
        
        
    def sigma_matrix_local (self,alpha,beta):
        """
        Fermions
        
        Outputs (sparse) matrix representation of operator sigma_alphabeta = f_alpha^dagger f_beta for current local Hilbert space
        """
        states = np.array(self.states_list)
        indices = np.arange(len(self.states_list))
        binaries = np.array(self.binaries_list)
        
        # Apply f_beta to states and remove previously unoccupied states
        states[:,beta] = states[:,beta] - 1
        
        remove = states[:,beta]>=0
        states = states[remove,:]
        indices = indices[remove]
        binaries = binaries[remove]
        
        # Apply f_alpha^dagger to states and remove previously occupied states
        states[:,alpha] = states[:,alpha] + 1
        
        remove = states[:,alpha]<=1
        states = states[remove,:]
        indices = indices[remove]
        binaries = binaries[remove]
        
        # Compute sign of operation
        if abs(alpha-beta)>1:
            if alpha<beta: sign = np.sum(states[:,alpha+1:beta],axis=1)
            else: sign = np.sum(states[:,beta+1:alpha],axis=1)
        else: sign = np.zeros(len(states), dtype=int)
            
        sign = (-1)**sign
        
        # Compute binaries of resulting states and substitute by their state number
        states = self.binary(states.transpose())
        states = np.searchsorted(self.binaries_list,states)
        
        #for ii in range(len(states)):
        #    print(indices[ii],states[ii],self.states_list[indices[ii]],self.states_list[states[ii]],sign[ii])
        
        # Compute sigma (sparse matrix)
        sigma = csc_matrix( (sign,(states,indices)) , shape=(self.localhilbertsize,self.localhilbertsize) )
        #sigma = np.zeros((self.localhilbertsize,self.localhilbertsize))
        #for ii in range(len(states)):
        #    sigma[states[ii],indices[ii]] = sign[ii]
        
        #print(sigma.toarray())
        
        return sigma
        
        
        
    def projector (self,n1,n2,ii):
        """
        Outputs (sparse) matrix representation of projector |n1><n2|
        """
        return sp.kron( sp.kron( sp.identity(self.localhilbertsize**ii) , self.projector_local(n1,n2) , format='csc'), sp.identity(self.localhilbertsize**(self.nsites-ii-1)) , format='csc')
        
        
    def projector_local (self,n1,n2):
        """
        Returns local projector |n1><n2|, where n1, n2 label states in the local Hilbert space
        """
        return csc_matrix( ([1],([n1],[n2])) , shape=(self.localhilbertsize,self.localhilbertsize))
        
    
    
    def get_statenumber(self,m):
        """
        Expects 1D array of integers specifying the levels that are occupied by an atom, i.e. | (m[0], m[1], m[2], ...) >  (ordered)
        Returns state number corresponding to that state.
        
        Note: Should add more exceptions and error handling.
        """
        b = np.zeros(self.numlevels, dtype=int)
        for ii in range(len(m)):
            if m[ii]<len(b):
                b[m[ii]] = 1
            else: print("Error: get_statenumber: Level numbers given larger than available.")
        if np.sum(b)!=self.filling: print("Error: get_statenumber: Too few levels given.")
        b = self.binary(b)
        return np.searchsorted(self.binaries_list,b)
        

    def output_stateslist(self):
        """Outputs file with the state corresponding to each statenumber.
        Move this to dipolar_system.py??"""
        ## Output list of states of local Hilbert space
        filename = '%s/liststates_fill%s_Ng%i_Ne%i_Ni%i.txt'%(param.outfolder, param.filling, param.deg_g, param.deg_e, param.deg_i)
        with open(filename,'w') as f:
            f.write('# Row 1: state number | Row 2: occupation excited | Row 3: occupation intermediate (if any) | Row 4: occupation ground.\n')
            f.write('# \n# \n')
            for ii in range(len(self.states_list)):
                f.write('%i\n'%(ii))
                np.savetxt(f,[list(self.states_list[ii][param.deg_g:param.deg_g+param.deg_e])],fmt='%i')
                if param.deg_i>0: np.savetxt(f,[list(self.states_list[ii][param.deg_g+param.deg_e:])],fmt='%i')
                np.savetxt(f,[list(self.states_list[ii][:param.deg_g])],fmt='%i')
                f.write('\n')
                
                #output_data = [ [ tt ] + list(self.states_list[tt]) for tt in range(len(self.states_list)) ]
                #np.savetxt(f,output_data,fmt='%i')
        
        ## Output |x> states of total Hilbert space in computational basis |y1,y2,y3,..>, where each y corresponds to a state of self.states_list
        if self.nsites>1:
            filename = '%s/globalstates_nsites%i_fill%s_Ng%i_Ne%i.txt'%(param.outfolder, self.nsites, param.filling, param.deg_g, param.deg_e)
            with open(filename,'w') as f:
                f.write('# Row 1: global state number |x> | Row 2: stat in computational basis |y1,y2,y3,..>\n')
                f.write('# \n# \n')
                for ii in range(self.hilbertsize):
                    f.write('%i\n'%(ii))
                    # Write hash in computational basis
                    y = ii
                    m = self.hilbertsize/self.localhilbertsize
                    index = int(y/m)
                    f.write('%i '%(index))
                    for jj in range(1,self.nsites):
                        y = y - index*m
                        m = m/self.localhilbertsize
                        index = int(y/m)
                        f.write('%i '%(index))
                    f.write('\n\n')
    


"""
numlevels = param.deg_e + param.deg_g
filling = param.filling

states_list = [np.zeros(numlevels, dtype=int)]
binaries_list = []
indices_list = []
temp = []



def binary(state):
    b = 0
    for ii in range(len(state)):
        b = b + state[ii]*2**ii
    return b
    

# Create list of states
for nn in range(1,filling+1):
    temp=states_list
    states_list=[]
    for ii in range(len(temp)):
        # Find last entry with a 1
        if nn>1: rindex = np.argwhere(temp[ii] == 1).max()
        else: rindex = 0
        
        for jj in range(rindex,numlevels):
            b = temp[ii] + 0
            b[jj] = b[jj] + 1
            if b[jj]<=1: states_list.append(b)
            
            
# Compute binaries and indices of all states
for ii in range(len(states_list)):
    binaries_list.append(binary(states_list[ii]))
indices_list = [ ii for ii in range(len(states_list))]

# Sort in increasing order
#for ii in range(len(states_list)):
#    print(states_list[ii],indices_list[ii],binaries_list[ii])
#print("\n")


# Order states in increasing order of binaries
ordering = sorted(range(len(binaries_list)),key=binaries_list.__getitem__)
states_list = [states_list[ii] for ii in ordering]
indices_list = [indices_list[ii] for ii in ordering]
binaries_list = [binaries_list[ii] for ii in ordering]


#for ii in range(len(states_list)):
#    print(states_list[ii],indices_list[ii],binaries_list[ii])

    


"""










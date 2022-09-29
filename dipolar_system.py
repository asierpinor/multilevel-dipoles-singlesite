
import math
from math import sqrt as sqrt

import numpy as np
from numpy.linalg import eig
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp

import os

import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
from scipy.special import comb
from scipy.integrate import complex_ode


from collections import defaultdict

import sys

import parameters as param
import hilbert_space

# Contains lattice parameters and functions for indexing

class Dipolar_System:
    
    """
    Class parameters: (self)
    
    dim
    Nlist
    Ntotal
    
    numlevels
    hilbertsize
    
    R_qaxis
    
    r_i
    f_ij
    g_ij
    
    basisVecs
    
    dipole_sph
    rabi
    
    linSuper
    
    
    memory_per_matrix
    
    """
    
    
    """
    Class functions:
    
    dummy_constants
    get_array_position
    get_indices
    kron_del
    fill_ri
    fill_dsph
    
    """
    
    def __init__(self):
        
        ### ---------------
        ###     LATTICE
        ### ---------------
        self.dim = 0
        self.Nlist = []
        
        if param.Nx>1: self.dim += 1; self.Nlist.append(param.Nx);
        if param.Ny>1: self.dim += 1; self.Nlist.append(param.Ny);
        if param.Nz>1: self.dim += 1; self.Nlist.append(param.Nz);
        if self.dim==0: self.dim = param.DIM; self.Nlist = [param.N for x in range(param.DIM)];
    
        self.Ntotal=1
        for ii in range(self.dim): self.Ntotal *= self.Nlist[ii];
        
        
        self.dummy_constants()
        self.R_qaxis = np.array( [[cos(param.phi_qa),-sin(param.phi_qa),0],[sin(param.phi_qa),cos(param.phi_qa),0],[0,0,1]] ) \
                        @ np.array( [[cos(param.theta_qa),0,sin(param.theta_qa)],[0,1,0],[-sin(param.theta_qa),0,cos(param.theta_qa)]] )
        self.R_kvector = [ np.array( [[cos(param.phi_k[ii]),-sin(param.phi_k[ii]),0],[sin(param.phi_k[ii]),cos(param.phi_k[ii]),0],[0,0,1]] ) \
                         @ np.array( [[cos(param.theta_k[ii]),0,sin(param.theta_k[ii])],[0,1,0],[-sin(param.theta_k[ii]),0,cos(param.theta_k[ii])]] ) for ii in range(self.nlasers) ]
                        
        self.ex=np.array([1,0,0])
        self.ey=np.array([0,1,0])
        self.ez=np.array([0,0,1])
        self.eplus=np.array([-1/sqrt(2),-1j/sqrt(2),0])
        self.eminus=np.array([1/sqrt(2),-1j/sqrt(2),0])
        
        
        self.kvec_laser = [ ( param.omega_laser[ii]/param.c * self.R_kvector[ii] @ self.ez.reshape((3,1)) ).reshape(3) for ii in range(self.nlasers)]        # Laser wave-vector
        pol_norm = [ sqrt( abs(param.pol_x[ii])**2 + abs(param.pol_y[ii])**2 ) for ii in range(self.nlasers)]
        self.polarization = [ ( self.R_kvector[ii] @ ( param.pol_x[ii]*self.ex + param.pol_y[ii]*self.ey ).reshape((3,1)) / pol_norm[ii] ).reshape(3) for ii in range(self.nlasers) ]     # Laser polarization
        
        
        ### ---------------
        ###     LEVELS
        ### ---------------
        self.levels_info = { 'Fe':param.Fe, 'deg_e':param.deg_e, 'Fg':param.Fg, 'deg_g':param.deg_g,\
                             'Fi':param.Fi, 'deg_i':param.deg_i, 'Fs':param.Fs, 'deg_s':param.deg_s }
        
        ### ---------------
        ###  SPACE VECTORS
        ### ---------------
        self.r_i = [np.zeros(3) for _ in range(self.Ntotal) ]
        self.fill_ri()
        
        ### ---------------------------------------------
        ###     DIPOLE MATRIX ELEMENTS
        ### ---------------------------------------------
        # Consider only d_{ab} with a in upper level (e) and b in lower level (g)
        ############## REMOVE self.dipole_sph and express everything in terms of self.dipole #############
        self.dipole_sph = [ [ np.zeros(3) for _ in range(param.deg_g) ] for _ in range(param.deg_e) ]
        self.fill_dsph()
        
        self.dipole = {}
        
        self.add_dsph_to_dictionary (param.Fe,param.Fg,param.deg_e,param.deg_g,'eg','ge')
        self.add_dsph_to_dictionary (param.Fi,param.Fg,param.deg_i,param.deg_g,'ig','gi')
        self.add_dsph_to_dictionary (param.Fs,param.Fg,param.deg_s,param.deg_g,'sg','gs')
        self.add_dsph_to_dictionary (param.Fe,param.Fi,param.deg_e,param.deg_i,'ei','ie')
        self.add_dsph_to_dictionary (param.Fs,param.Fi,param.deg_s,param.deg_i,'si','is')
        self.add_dsph_to_dictionary (param.Fs,param.Fe,param.deg_s,param.deg_e,'se','es')
        
        #print(self.dipole['ig'])
        #for bb in range(param.deg_g):
        #    for aa in range(param.deg_e):
        #        print( abs( self.dipole['eg'][aa][bb]-self.dipole_sph[aa][bb] ) )
        #        print( abs( self.dipole['ig'][aa][bb]-self.dipole_sph[aa][bb] ) )
                
        
        ### ---------------
        ###  BASIS VECTORS
        ### ---------------
        # Mapping between e-g numbering and a flattened-out level numbering, which will be used for Hilbert space.
        # Ordering of atomic levels: g to e, left to right
        self.ud_to_level = { 'g': [ bb for bb in range(param.deg_g) ] ,\
                             'e': [ param.deg_g+aa for aa in range(param.deg_e) ] ,\
                             'i': [ param.deg_g+param.deg_e+cc for cc in range(param.deg_i) ] }
        self.level_to_ud = []
        for nn in range(self.numlevels):
            if nn<param.deg_g: self.level_to_ud.append(['g',nn])
            if nn>=param.deg_g and nn<param.deg_g+param.deg_e: self.level_to_ud.append(['e',nn-param.deg_g])
            if nn>=param.deg_g+param.deg_e: self.level_to_ud.append(['i',nn-param.deg_g-param.deg_e])
            
        self.hspace = hilbert_space.Hilbert_Space(self.numlevels,param.filling,self.Ntotal)
        
        ### ---------------
        ###     MEMORY
        ### ---------------
        # Real
        self.memory_full_Linblad = self.hspace.hilbertsize**4 * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        self.memory_full_Hamiltonian = self.hspace.hilbertsize**2 * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        self.memory_wavefunction = self.hspace.hilbertsize * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        print("\nMemory full Linblad: %g Gb."%(self.memory_full_Linblad))
        print("Memory full Hamiltonian: %g Gb."%(self.memory_full_Hamiltonian))
        print("Memory wave-function: %g Gb."%(self.memory_wavefunction))
        
        # Estimated
        self.memory_estimate_sparse_Linblad = self.Ntotal**2 * (3*max([param.deg_e,param.deg_g]))**2 \
                                             * (comb(self.numlevels-2,param.filling-1)*self.hspace.hilbertsize/self.hspace.localhilbertsize) * self.hspace.hilbertsize * 4 * 8 /1024**3
        # Number of terms in sum over {alpha,beta,alpha',beta',ii,jj} x nonzero entries in sigma_alphabeta^ii x tensor product with "1" x (2 position integers + 1 complex number) x 8 bytes/double / Bytes per Gb
        self.memory_estimate_sparse_Hamiltonian = self.Ntotal**2 * (3*max([param.deg_e,param.deg_g]))**2 \
                                             * (comb(self.numlevels-2,param.filling-1)*self.hspace.hilbertsize/self.hspace.localhilbertsize) * 4 * 8 /1024**3
        print("Estimated memory of sparse Linblad: %g Gb."%(self.memory_estimate_sparse_Linblad))
        print("Estimated memory of sparse Hamiltonian: %g Gb."%(self.memory_estimate_sparse_Hamiltonian))
        
        
        
        
        
    def decide_if_timeDep (self,phase):
        """
        Set self.timeDep to True or False.
        The boolean variable self.timeDep answers the question 'Is the Linbladian explicitly time-dependent?'
        """
        if param.switchlaser[phase]==False:
            self.timeDep = False
        else:
            if self.nlasers==0: self.timeDep = False
            if self.nlasers==1:
                if param.omegaR == param.omega_laser[0]: self.timeDep = False
                else:
                    self.timeDep = True
                    print("\nWARNING/dipolar_system: The problem is time-dependent, but can be avoided by choosing param.omegaR appropriately, unless Raman is used as well.\n")
            if self.nlasers>1:
                if param.detuning.count(param.detuning[0]) == len(param.detuning): self.timeDep = False
                else:
                    self.timeDep = True
                    print("\nNOTE/dipolar_system: Problem is time-dependent.\n")
        
        
        
        
    def dummy_constants (self):
        
        self.numlevels = param.deg_e + param.deg_g + param.deg_i      # Total number of internal electronic states per atom
        self.nlasers = len(param.rabi_coupling)


    
    def get_array_position (self,indices):
        """Returns array position in row-order of a given lattice site (i1,i2,i3,...,in)"""
        if len(indices)!=self.dim: print("\nERROR/get_array_position: indices given to get_array_position have wrong length\n")
        
        array_position = indices[0]
        for ii in range(1,self.dim):
            array_position = array_position*self.Nlist[ii] + indices[ii]
        return array_position
    
    

    def get_indices (self,n):
        """Returns lattice indices (i1,i2,...) for a given array position"""
        indices = []
        temp = 0
        rest = n
        block = self.Ntotal
    
        while temp<self.dim:
            block = int( block/self.Nlist[temp] )
            indices.append(rest//block)     # should be able to do this more efficiently
            rest -= indices[temp]*block
            temp += 1
            
        return indices
        
        
        
    
    def kron_del(self,a1,a2):
        """Kroenecker delta"""
        if a1==a2:
            return 1
        else:
            return 0
            
    
    def fill_ri (self):
        """Fill out matrix of atom positions r_i"""
        if (param.geometry=='lattice'):
            """Lattice oriented along x in 1D, along xy in 2D, and along xyz in 3D
            NOTE: Not using periodic boundary conditions."""
            for r1 in range(self.Ntotal):
                ind1 = self.get_indices(r1)
                while len(ind1)<3: ind1.append(0)
                self.r_i[r1] = np.array(ind1) * param.latsp
        #if (param.geometry=='cavity'):
            # Do nothing
   
    
    
    def fill_rabi (self):
        """Compute rabi coupling"""
        for nu in range(self.nlasers):
            for aa in range(param.deg_e):
                for bb in range(param.deg_g):
                    self.rabi[nu][aa,bb] = param.rabi_coupling[nu] * np.dot( self.polarization[nu], self.dipole_sph[aa][bb] )
                    
                    
                    
    def fill_effrabi_raman (self):
        """Compute effective rabi coupling of Raman transition"""
        self.fill_dsph_intermediate()
        self.effrabi_raman = np.zeros((param.deg_e,param.deg_g),dtype='complex')
        for aa in range(param.deg_e):
            for bb in range(param.deg_g):
                for ss in range(param.deg_s):
                    self.effrabi_raman[aa,bb] = self.effrabi_raman[aa,bb] + param.effrabi_coupling_raman \
                                                                            * np.dot( param.pol1_raman, self.dipole_sph_sg[ss][bb] ) \
                                                                            * ( np.dot( param.pol2_raman, self.dipole_sph_se[ss][aa] ).conj() )
        
        
    
    
    def fill_dsph (self):
        
        # Define dipole matrix elements in quantization axis coordinate system
        options_dsph = defaultdict( self.dsph_two_level )

        options_dsph = { 'two_level': self.dsph_two_level, 'Sr88': self.dsph_Sr88, 'Sr87': self.dsph_Sr87, 'Yb171': self.dsph_Yb171, 'generic': self.dsph_generic }

        options_dsph[param.dipole_structure]()

        # Rotate with respect to lattice
        for aa in range(param.deg_e):
            for bb in range(param.deg_g):
                #print( np.array( [[1,0,0],[0,1,0],[0,0,1]] )  @ np.resize(self.dipole_sph[aa][bb],(3,1)) )
                self.dipole_sph[aa][bb] = (self.R_qaxis @ np.reshape(self.dipole_sph[aa][bb],(3,1)) ).reshape(3)
                
    
    def add_dsph_to_dictionary (self,Fa,Fb,deg_a,deg_b,key,keycomplex):
        """Adds the spherical dipole matrix elements d_{ab} connecting states a and b to the self.dipole dictionary."""
        dipole_temp_ab = [ [ np.zeros(3) for _ in range(deg_b) ] for _ in range(deg_a) ]
        dipole_temp_ba = [ [ np.zeros(3) for _ in range(deg_a) ] for _ in range(deg_b) ]
        
        # Read in dipole matrix elements
        if abs(Fa-Fb)>1.01:
            print('\nERROR: Inconsistent choice of dipole_structure and number of levels for intermediate state\n')
            return 0
        else:
            mas = [ -Fa+ma for ma in range(deg_a) ]
            mbs = [ -Fb+mb for mb in range(deg_b) ]
            for mb in range(len(mbs)):
                for ma in range(len(mas)):
                    if abs(mas[ma]-mbs[mb])<1.01:
                        filename = 'dsph_Fg%g_Fe%g_mg%g_me%g'%(Fb,Fa,mbs[mb],mas[ma])
                        dipole_temp_ab[ma][mb] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
                        
        # Rotate with respect to lattice
        for bb in range(deg_b):
            for aa in range(deg_a):
                dipole_temp_ab[aa][bb] = (self.R_qaxis @ np.reshape(dipole_temp_ab[aa][bb],(3,1)) ).reshape(3)
                
        # Complex conjugates
        # NOTE: the spherical dipole moment does not have the property d_{eg}=d_{ge}^*.
        #       What we are computing here is just the complex conjugate of d_{eg}, NOT the spherical component d_{ge}
        #       By defining it in this way we assure I think that the Rabi frequency does not change (up to signs) when defining laser as "eg" vs "ge".
        dipole_temp_ba = [ [ dipole_temp_ab[aa][bb].conj() for aa in range(deg_a) ] for bb in range(deg_b) ]
        
        # Define dictionary
        # Correct one
        self.dipole[key] = dipole_temp_ab
        # Dummy one
        self.dipole[keycomplex] = dipole_temp_ba
        

                
                
    def fill_dsph_intermediate (self):
        """Reads in dipole matrix elements connecting e, g with intermediate state s."""
        self.dipole_sph_sg = [ [ np.zeros(3) for _ in range(param.deg_g) ] for _ in range(param.deg_s) ]
        self.dipole_sph_se = [ [ np.zeros(3) for _ in range(param.deg_e) ] for _ in range(param.deg_s) ]
        
        # Read in dipole matrix elements
        if abs(param.Fs-param.Fg)>1.01 or abs(param.Fs-param.Fe)>1.01 or param.deg_s>2*param.Fs+1:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels for intermediate state')
            return 0
        else:
            mes = [ -param.Fe+me for me in range(param.deg_e) ]
            mgs = [ -param.Fg+mg for mg in range(param.deg_g) ]
            mss = [ -param.Fs+ms for ms in range(param.deg_s) ]
            for mg in range(len(mgs)):
                for ms in range(len(mss)):
                    if abs(mss[ms]-mgs[mg])<1.01:
                        filename = 'dsph_Fg%g_Fe%g_mg%g_me%g'%(param.Fg,param.Fs,mgs[mg],mss[ms])
                        self.dipole_sph_sg[ms][mg] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
            for me in range(len(mes)):
                for ms in range(len(mss)):
                    if abs(mss[ms]-mes[me])<1.01:
                        filename = 'dsph_Fg%g_Fe%g_mg%g_me%g'%(param.Fe,param.Fs,mes[me],mss[ms])
                        self.dipole_sph_se[ms][me] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)

        # Rotate with respect to lattice
        for aa in range(param.deg_s):
            for bb in range(param.deg_g):
                self.dipole_sph_sg[aa][bb] = (self.R_qaxis @ np.reshape(self.dipole_sph_sg[aa][bb],(3,1)) ).reshape(3)
            for bb in range(param.deg_e):
                self.dipole_sph_se[aa][bb] = (self.R_qaxis @ np.reshape(self.dipole_sph_se[aa][bb],(3,1)) ).reshape(3)
        
        
            
            
            
    def dsph_two_level (self):
        """Two-level system with dipole aligned along quantization axis"""
        if param.deg_e!=1 or param.deg_g!=1:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels')
            return 0
        else:
            self.dipole_sph[0][0] = np.array([0,0,1])

        
        
    def dsph_Sr88 (self):
        """Level structure of 1S_0 (ground) -> 1P_1/3P_1 (excited) with Jg=0 and Je=1."""
        if param.deg_e!=3 or param.deg_g!=1:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels')
            return 0
        else:
            mes = [-1,0,1]
            for me in range(len(mes)):
                filename = 'dsph_Fg0_Fe1_mg0_me%g'%mes[me]
                self.dipole_sph[me][0] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
        #print(os.getcwd())


    def dsph_Sr87 (self):
        """Level structure of 1S_0 (ground) -> 3P_0/3P_1 (excited) with Fg=9/2 and Fe=7/2,9/2,11/2.
        The number of levels to use can in principle differ from the actual total number of Zeeman states.
        States are labeled from left to right, i.e. starting at -F."""
        if param.deg_e>2*param.Fe+1 or param.deg_g>10:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels')
            return 0
        else:
            mes = [ -param.Fe+me for me in range(param.deg_e) ]
            mgs = [ -9/2+mg for mg in range(param.deg_g) ]
            for mg in range(len(mgs)):
                for me in range(len(mes)):
                    if abs(mes[me]-mgs[mg])<1.01:
                        filename = 'dsph_Fg4.5_Fe%g_mg%g_me%g'%(param.Fe,mgs[mg],mes[me])
                        self.dipole_sph[me][mg] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
    
    
    def dsph_Yb171 (self):
        """Level structure of 1S_0 (ground) -> 3P0/3P1 (excited) with Fg=1/2 and Fe=1/2."""
        if param.deg_e!=2 or param.deg_g!=2:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels')
            return 0
        else:
            mes = [-1/2,1/2]
            mgs = [-1/2,1/2]
            for mg in range(len(mgs)):
                for me in range(len(mes)):
                    filename = 'dsph_Fg0.5_Fe0.5_mg%g_me%g'%(mgs[mg],mes[me])
                    self.dipole_sph[me][mg] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
                
                
    def dsph_generic (self):
        """Level structure of with Fg and Fe given in parameters."""
        if abs(param.Fe-param.Fg)>1.01 or param.deg_e>2*param.Fe+1 or param.deg_g>2*param.Fg+1:
            print('ERROR: Inconsistent choice of dipole_structure and number of levels')
            return 0
        else:
            mes = [ -param.Fe+me for me in range(param.deg_e) ]
            mgs = [ -param.Fg+mg for mg in range(param.deg_g) ]
            for mg in range(len(mgs)):
                for me in range(len(mes)):
                    if abs(mes[me]-mgs[mg])<1.01:
                        filename = 'dsph_Fg%g_Fe%g_mg%g_me%g'%(param.Fg,param.Fe,mgs[mg],mes[me])
                        self.dipole_sph[me][mg] = np.fromfile(param.folder_dsph+filename,dtype=np.complex)
                        
    
    
    def f_ij (self,i1,i2,alpha1,beta1,alpha2,beta2):
        """Function returns value of dipolar dissipative coupling f between i1 and i2 atoms.
        Assumes alphas are from 'e', betas from 'g'.
        For i1==i2 it returns the limit r->0 of the function, essentially Gamma/2"""
        if i1==i2:
            return param.Gamma/2 * np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )
        else:
            k = param.k0
            rvec = self.r_i[i1]-self.r_i[i2]
            r = math.sqrt( sum( rvec**2 ) )
            rvec = rvec/r
            
            diptemp1 = np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )\
                        - np.dot( rvec, self.dipole_sph[alpha1][beta1] ) * np.dot( rvec, np.conj(self.dipole_sph[alpha2][beta2]) )
            diptemp3 = np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )\
                        - 3 * np.dot( rvec, self.dipole_sph[alpha1][beta1] ) * np.dot( rvec, np.conj(self.dipole_sph[alpha2][beta2]) )
            
            return 3*param.Gamma/4 * ( diptemp1 * sin(k*r)/(k*r) + diptemp3 * ( cos(k*r)/(k*r)**2 - sin(k*r)/(k*r)**3 ) )
            
            
            
    def g_ij (self,i1,i2,alpha1,beta1,alpha2,beta2):
        """Function returns value of dipolar coherent coupling g between i1 and i2 atoms.
        Assumes alphas are from 'e', betas from 'g'.
        For i1==i2 it returns the limit r->0 of the function, essentially Gamma/2"""
        if i1==i2:
            k = param.k0
            
            diptemp3 = np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )\
                        - 3 * np.dot( self.ez , self.dipole_sph[alpha1][beta1] ) * np.dot( self.ez , np.conj(self.dipole_sph[alpha2][beta2]) )
            
            return - 3*param.Gamma/4 * param.onsite_prefactor * diptemp3
        else:
            k = param.k0
            rvec = self.r_i[i1]-self.r_i[i2]
            r = math.sqrt( sum( rvec**2 ) )
            rvec = rvec/r
            
            diptemp1 = np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )\
                        - np.dot( rvec, self.dipole_sph[alpha1][beta1] ) * np.dot( rvec, np.conj(self.dipole_sph[alpha2][beta2]) )
            diptemp3 = np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )\
                        - 3 * np.dot( rvec, self.dipole_sph[alpha1][beta1] ) * np.dot( rvec, np.conj(self.dipole_sph[alpha2][beta2]) )
                        
            #print(rvec,self.dipole_sph[alpha1][beta1],np.dot( rvec, self.dipole_sph[alpha1][beta1] ))
            
            return 3*param.Gamma/4 * ( diptemp1 * cos(k*r)/(k*r) - diptemp3 * ( sin(k*r)/(k*r)**2 + cos(k*r)/(k*r)**3 ) )
    





    

####################################################################

#######                INITIAL CONDITIONS                ###########

####################################################################
        
    def choose_initial_condition (self,cIC=param.cIC):
        """Initialize density matrix to chosen initial condition."""
        optionsIC = defaultdict( self.IC_initialstate )
        optionsIC = { 'initialstate': self.IC_initialstate,
                      'dark': self.IC_dark,
                      'test': self.IC_test}
        optionsIC[cIC]()
        
        #self.rho2 = self.rho
        
        self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)        # 'C': row-major, "F": column-major
        
    
    def IC_initialstate (self):
        """
        Initialize all sites with the atoms in the state specified by param.initialstate.
        """
        if len(param.initialstate)!=param.filling: print("\nWarning: Length of initialstate doesn't match filling.\n")
        
        occlevels = []
        for ii in range(len(param.initialstate)):
            occlevels.append( self.ud_to_level[ param.initialstate[ii][0] ][ int(param.initialstate[ii][1:]) ] )    
        occupied_state = self.hspace.get_statenumber( occlevels )
        
        rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
        rhoi[occupied_state,occupied_state] = 1
        self.rho = rhoi
        for ii in range(1,self.Ntotal):
            self.rho = np.kron( self.rho , rhoi )
        
        #rhoi2 = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
        #othergs = abs(1-occupied_state)
        #rhoi2[othergs,othergs] = 1
        #self.rho = np.kron(self.rho, rhoi2 )
        
            
            
    def IC_dark (self):
        """Initialize all sites with the atoms in the dark state of the four level system with Sr87 dipole elements.
        Only valid for 2 atoms/site."""
        if param.filling!=2: print("\nWarning: Chosen IC not thought for filling!=2.\n")
        rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
        state_e0g1 = self.hspace.get_statenumber( [ self.ud_to_level['e'][0] , self.ud_to_level['g'][1] ] )
        state_e1g0 = self.hspace.get_statenumber( [ self.ud_to_level['e'][1] , self.ud_to_level['g'][0] ] )
        rhoi[state_e0g1,state_e0g1] = 49/(130)
        rhoi[state_e1g0,state_e1g0] = 81/130
        rhoi[state_e0g1,state_e1g0] = 63/130
        rhoi[state_e1g0,state_e0g1] = 63/130
        self.rho = rhoi
        for ii in range(1,self.Ntotal):
            self.rho = np.kron( self.rho , rhoi )
            
            
    def IC_test (self):
        """Initialize all sites with the atoms in the state chosen below."""
        if param.filling!=2: print("\nWarning: Chosen IC not thought for filling!=2.\n")
        if param.deg_e!=6 or param.deg_g!=4: print("\nWarning: Chosen IC not thought for this internal level structure.\n")
        rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
        
        state = np.zeros((self.hspace.localhilbertsize,1))
        
        # Dark: 3/2 -> 5/2, M=3
        state_1 = self.hspace.get_statenumber( [ self.ud_to_level['e'][5] , self.ud_to_level['g'][2] ] )
        state_2 = self.hspace.get_statenumber( [ self.ud_to_level['e'][4] , self.ud_to_level['g'][3] ] )
        state[state_1,0] = sqrt(3/2)/2
        state[state_2,0] = sqrt(5/2)/2
        
        # Dark: 3/2 -> 5/2, M=2
        #state_1 = self.hspace.get_statenumber( [ self.ud_to_level['e'][5] , self.ud_to_level['g'][1] ] )
        #state_2 = self.hspace.get_statenumber( [ self.ud_to_level['e'][4] , self.ud_to_level['g'][2] ] )
        #state_3 = self.hspace.get_statenumber( [ self.ud_to_level['e'][3] , self.ud_to_level['g'][3] ] )
        #state[state_1,0] = sqrt(3/7)/2
        #state[state_2,0] = sqrt(15/7)/2
        #state[state_3,0] = sqrt(5/14)
        
        # Dark: 3/2 -> 5/2, M=1
        #state_1 = self.hspace.get_statenumber( [ self.ud_to_level['e'][5] , self.ud_to_level['g'][0] ] )
        #state_2 = self.hspace.get_statenumber( [ self.ud_to_level['e'][4] , self.ud_to_level['g'][1] ] )
        #state_3 = self.hspace.get_statenumber( [ self.ud_to_level['e'][3] , self.ud_to_level['g'][2] ] )
        #state_4 = self.hspace.get_statenumber( [ self.ud_to_level['e'][2] , self.ud_to_level['g'][3] ] )
        #state[state_1,0] = 1/(2*sqrt(14))
        #state[state_2,0] = sqrt(15/14)/2
        #state[state_3,0] = sqrt(15/7)/2
        #state[state_4,0] = sqrt(5/7)/2
        
        # Dark: 3/2 -> 5/2, M=0
        #state_1 = self.hspace.get_statenumber( [ self.ud_to_level['e'][4] , self.ud_to_level['g'][0] ] )
        #state_2 = self.hspace.get_statenumber( [ self.ud_to_level['e'][3] , self.ud_to_level['g'][1] ] )
        #state_3 = self.hspace.get_statenumber( [ self.ud_to_level['e'][2] , self.ud_to_level['g'][2] ] )
        #state_4 = self.hspace.get_statenumber( [ self.ud_to_level['e'][1] , self.ud_to_level['g'][3] ] )
        #state[state_1,0] = 1/sqrt(14)
        #state[state_2,0] = sqrt(3/7)
        #state[state_3,0] = sqrt(3/7)
        #state[state_4,0] = 1/sqrt(14)
        
        rhoi = state @ state.conj().T
        
        print(rhoi)

        #rhoi[state_1,state_1] = c1*c1
        #rhoi[state_2,state_2] = c2*c2
        #rhoi[state_3,state_3] = c3*c3
        #rhoi[state_4,state_4] = c4*c4
        
        
        self.rho = rhoi
        for ii in range(1,self.Ntotal):
            self.rho = np.kron( self.rho , rhoi )
            
     
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
####################################################################

######                LINBLAD SUPEROPERATOR                #########

####################################################################
        
    def save_partial_HamLin (self):
        """
        Saves separately the Hamiltonians and Linbladians of the dipolar, rabi and energies parts.
        """
        self.linSuper_dipoles = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        self.ham_eff_dipoles = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        if param.geometry!='cavity': self.define_HL_dipolar()
        
        self.linSuper_cavity = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        self.ham_eff_cavity = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        if param.geometry=='cavity': self.define_HL_cavity()
        
        # Do the same with other Hams
        self.define_H_rabi()
        #self.define_H_rabi_old()
        self.define_H_effrabi_raman()
        #self.define_H_effrabi_raman_old()
        self.define_H_energies()
    
        
    def define_linblad_superop (self,phase):
        """
        Computes Linblad superoperator and effective Hamiltonian.
        For time-dependent problems it saves Linblads to speed up computation of time-dependent Linblad during dynamics.
        """
        one = sp.identity(self.hspace.hilbertsize,format='csc')
        self.linSuper = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        self.ham_eff_total = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Energies, dipoles, cavity Hamiltonians
        self.ham_eff_total = self.H_energies + self.ham_eff_dipoles + self.ham_eff_cavity
        
        # Rabi Hamiltonian
        if param.switchlaser[phase]==True:
            if self.timeDep == False:
                # Add to total Hamiltonian
                for nu in range(self.nlasers):
                    self.ham_eff_total = self.ham_eff_total + self.H_rabi[nu] + self.H_rabi[nu].conj().T
            if self.timeDep == True:
                # Prepare Linblads for time-dependent part (without -1j factor)
                self.linSuper_timeDep = [ sp.kron(self.H_rabi[nu],one,format='csc') - sp.kron(one,self.H_rabi[nu].T,format='csc') for nu in range(self.nlasers) ]
                
        # Effective Rabi Hamiltonian, Raman transition
        if param.switchlaser_raman[phase]==True:
            self.ham_eff_total = self.ham_eff_total + self.H_effrabi_raman
                
        # Total Linblad (time-independent part)
        self.linSuper = self.linSuper_dipoles + self.linSuper_cavity - 1j * ( sp.kron(self.ham_eff_total,one,format='csc') - sp.kron(one,self.ham_eff_total.conj(),format='csc') )
            
        
        
        
    def compute_memory (self):
        """Compute memory and sparsity of Linblad and Hamiltonian"""
        self.memory_sparse_Linblad = (self.linSuper.data.nbytes + self.linSuper.indptr.nbytes + self.linSuper.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        self.memory_sparse_Hamiltonian = (self.ham_eff_total.data.nbytes + self.ham_eff_total.indptr.nbytes + self.ham_eff_total.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        self.sparsity_Linblad = self.linSuper.nnz / np.prod(self.linSuper.shape)
        self.sparsity_Hamiltonian = self.ham_eff_total.nnz / np.prod(self.ham_eff_total.shape)
        print("\nMemory for sparse Linblad: %g Gb."%(self.memory_sparse_Linblad))
        print("Memory for sparse Hamiltonian: %g Gb."%(self.memory_sparse_Hamiltonian))
        print("Sparsity of Linblad: %g"%(self.sparsity_Linblad))
        print("Sparsity of Hamiltonian: %g"%(self.sparsity_Hamiltonian))
        
        #sys.getsizeof(self.ham_eff_total)
        
        
    
    
    def define_HL_dipolar (self):
        """Saves Hamiltonian and Linbladian part of dipolar interactions."""
        """NOTE: I SHOULD CHANGE THE BUILD UP TO LIL_MATRIX, IT'S MUCH FASTER! """
        # Incoherent dipole part
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                
                                if self.f_ij(ii,jj,a1,b1,a2,b2)!=0:
                                        
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.ud_to_level['e'][a1] , self.ud_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.ud_to_level['e'][a2] , self.ud_to_level['g'][b2] , jj )
                                
                                    #if a1==a1 and b1==b2:
                                    self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T
                                
                                    self.linSuper_dipoles = self.linSuper_dipoles + 2 * self.f_ij(ii,jj,a1,b1,a2,b2) * sp.kron( sigma_a2b2_j.conj().T , np.transpose(sigma_a1b1_i) , format='csc' )
                                
                                    #self.linSuper = self.linSuper - self.f_ij(ii,jj,a1,b1,a2,b2) * ( sp.kron( sigma_a1b1_i@sigma_a2b2_j.conj().T , one , format='csc' ) \
                                    #                                                                + sp.kron( one , np.transpose(sigma_a1b1_i@sigma_a2b2_j.conj().T) , format='csc' )\
                                    #                                                                - 2 * sp.kron( sigma_a2b2_j.conj().T , np.transpose(sigma_a1b1_i) , format='csc' ) )
        
        # Coherent dipole part
        self.ham_eff_dipoles = self.ham_eff_dipoles + self.define_H_dipole()
        
        
        
    def define_HL_cavity (self):
        """Saves Hamiltonian and Linbladian part of cavity-mediated interactions."""
        # Incoherent dipole part
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                
                                if self.f_ij(ii,ii,a1,b1,a2,b2)!=0:     # all-to-all interactions, therefore set ii=jj
                                        
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.ud_to_level['e'][a1] , self.ud_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.ud_to_level['e'][a2] , self.ud_to_level['g'][b2] , jj )
                                    
                                    prefactor = param.g_inc * ( np.dot( param.cavpol_z, self.dipole_sph[a1][b1] ) * np.dot( param.cavpol_z, self.dipole_sph[a2][b2] ) \
                                                              + np.dot( param.cavpol_x, self.dipole_sph[a1][b1] ) * np.dot( param.cavpol_x, self.dipole_sph[a2][b2] )  )
                                
                                    self.ham_eff_cavity = self.ham_eff_cavity - 1j * prefactor * sigma_a1b1_i@sigma_a2b2_j.conj().T
                                
                                    self.linSuper_cavity = self.linSuper_cavity + 2 * prefactor * sp.kron( sigma_a2b2_j.conj().T , np.transpose(sigma_a1b1_i) , format='csc' )
                                    
                                    # np.dot( param.pol1_raman, self.dipole_sph_sg[ss][bb] )
                                    # param.Gamma/2 * np.dot( self.dipole_sph[alpha1][beta1], np.conj(self.dipole_sph[alpha2][beta2]) )
        
        # Coherent dipole part (not added for now, assume cavity is on resonance)
        #self.ham_eff_cavity = self.ham_eff_cavity #+ self.define_H_dipole()
        
        
    def define_H_energies (self):
        """Saves Hamiltonian of energy levels including Zeeman shifts. In rotating frame of omegaR."""
        """NOTE: I SHOULD CHANGE THE BUILD UP TO LIL_MATRIX, IT'S MUCH FASTER!
                AND ALSO DIRECTLY ACCESS THE NONZERO ENTRIES DIRECTLY WITH [.,.] INSTEAD OF +="""
        self.H_energies = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        for ii in range(self.Ntotal):
            for aa in range(param.deg_e):
                sigma_aa_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['e'][aa] , ii )
                #self.H_energies = self.H_energies + ((param.omega0+aa*param.zeeman_e)-param.omegaR) * sigma_aa_i
                self.H_energies = self.H_energies + ((param.omega0+(-param.Fe+aa)*param.zeeman_e)-param.omegaR) * sigma_aa_i
                
            for bb in range(param.deg_g):
                sigma_bb_i = self.hspace.sigma_matrix( self.ud_to_level['g'][bb] , self.ud_to_level['g'][bb] , ii )
                #self.H_energies = self.H_energies + (bb*param.zeeman_g) * sigma_bb_i
                self.H_energies = self.H_energies + ((-param.Fg+bb)*param.zeeman_g) * sigma_bb_i
        
        # Energy shift of n_e=2 
        if param.epsilon_ne2!=0:
            for ii in range(self.Ntotal):
                projection_i = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
                for aa in range(param.deg_e): projection_i = projection_i + self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['e'][aa] , ii )
                for aa in range(param.deg_i): projection_i = projection_i + self.hspace.sigma_matrix( self.ud_to_level['i'][aa] , self.ud_to_level['i'][aa] , ii )
                self.H_energies = self.H_energies + param.epsilon_ne2 * projection_i @ ( projection_i - sp.identity(self.hspace.hilbertsize) )
        
        return self.H_energies
    
        
    
    
    def define_H_rabi (self):
        """Saves Hamiltonian of Rabi coupling, only first summand."""
        self.H_rabi = [ csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize)) for ii in range(self.nlasers) ]
        
        for nu in range(self.nlasers):
            # Select levels
            levels = param.levels_laser[nu]
            level_a = levels[0]
            level_b = levels[1]
            deg_a = self.levels_info['deg_'+level_a]
            deg_b = self.levels_info['deg_'+level_b]
            
            # Check that levels are correct
            if (level_a!='e' and level_a!='g' and level_a!='i') or (level_b!='e' and level_b!='g' and level_b!='i') or level_a==level_b:
                print('\nERROR/define_H_rabi: Levels chosen not supported.\n')
            
            # Compute Rabi coupling
            rabi = np.zeros((deg_a,deg_b),dtype='complex')
            for aa in range(deg_a):
                for bb in range(deg_b):
                    rabi[aa,bb] = param.rabi_coupling[nu] * np.dot( self.polarization[nu], self.dipole[levels][aa][bb] )
            
            # Add to Hamiltonian
            for ii in range(self.Ntotal):
                for aa in range(deg_a):
                    for bb in range(deg_b):
                        sigma_ab_i = self.hspace.sigma_matrix( self.ud_to_level[level_a][aa] , self.ud_to_level[level_b][bb] , ii )
                        self.H_rabi[nu] = self.H_rabi[nu] - exp( 1j * np.dot(self.kvec_laser[nu],self.r_i[ii]) ) * rabi[aa,bb] * sigma_ab_i
                            
    
    
    def define_H_rabi_old (self):
        """Saves Hamiltonian of Rabi coupling, only first summand."""
        self.H_rabi = [ csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize)) for ii in range(self.nlasers) ]
        
        self.rabi = [ np.zeros((param.deg_e,param.deg_g),dtype='complex') for ii in range(self.nlasers)]
        self.fill_rabi()
        
        for ii in range(self.Ntotal):
            for aa in range(param.deg_e):
                for bb in range(param.deg_g):
                    
                    sigma_ab_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['g'][bb] , ii )
                    for nu in range(self.nlasers):
                        self.H_rabi[nu] = self.H_rabi[nu] - exp( 1j * np.dot(self.kvec_laser[nu],self.r_i[ii]) ) * self.rabi[nu][aa,bb] * sigma_ab_i
                        
            
        
        #print("Coupling G to B:")
        #print( 1/sqrt(2)*np.array([1,0,0,0,0,0])@(self.H_rabi[0]+self.H_rabi[0].conj().T)@np.array([[0],[0],[1],[1],[0],[0]]) )
        #print("Couplilng B to I:")
        #print( 1/sqrt(2)*np.array([0,0,1,1,0,0])@(self.H_rabi[0]+self.H_rabi[0].conj().T)@np.array([[0],[0],[0],[0],[0],[1]]) )
                        
            
    
    def define_H_effrabi_raman (self):
        """Saves Hamiltonian of 2-photon Raman transition."""
        self.H_effrabi_raman = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        # Select levels
        levels = param.levels_Raman
        level_a = levels[0]
        level_b = levels[1]
        level_c = levels[2]
        deg_a = self.levels_info['deg_'+level_a]
        deg_b = self.levels_info['deg_'+level_b]
        deg_c = self.levels_info['deg_'+level_c]
        
        # Check that levels are correct. EDIT: can probably be written more elegantly
        if (level_a!='e' and level_a!='g' and level_a!='i') or (level_b!='e' and level_b!='g' and level_b!='i') or level_a==level_b or level_a==level_c or level_b==level_c or (level_c!='e' and level_c!='g' and level_c!='i' and level_c!='s'):
            print('\nERROR/define_H_effrabi_raman: Levels chosen not supported.\n')
            
        # Compute Rabi coupling
        effrabi_raman = np.zeros((deg_a,deg_b),dtype='complex')
        for aa in range(deg_a):
            for bb in range(deg_b):
                for cc in range(deg_c):
                    effrabi_raman[aa,bb] = effrabi_raman[aa,bb] + param.effrabi_coupling_raman \
                                                                            * np.dot( param.pol1_raman, self.dipole[level_c+level_b][cc][bb] ) \
                                                                            * ( np.dot( param.pol2_raman, self.dipole[level_c+level_a][cc][aa] ).conj() )
                                                                            
        # Add to Hamiltonian
        for ii in range(self.Ntotal):
            for aa in range(deg_a):
                for bb in range(deg_b):
                    
                    sigma_ab_i = self.hspace.sigma_matrix( self.ud_to_level[level_a][aa] , self.ud_to_level[level_b][bb] , ii )
                    self.H_effrabi_raman = self.H_effrabi_raman + effrabi_raman[aa,bb] * sigma_ab_i
                    
        self.H_effrabi_raman = self.H_effrabi_raman + self.H_effrabi_raman.conj().T
    
    
        
    def define_H_effrabi_raman_old (self):
        """Saves Hamiltonian of 2-photon Raman transition."""
        self.H_effrabi_raman = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        self.fill_effrabi_raman()
        
        for ii in range(self.Ntotal):
            for aa in range(param.deg_e):
                for bb in range(param.deg_g):
                    
                    sigma_ab_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['g'][bb] , ii )
                    self.H_effrabi_raman = self.H_effrabi_raman + self.effrabi_raman[aa,bb] * sigma_ab_i
                    
        self.H_effrabi_raman = self.H_effrabi_raman + self.H_effrabi_raman.conj().T
        
        #print("Effective Rabi")
        #print(self.effrabi_raman)
        #print(self.H_effrabi_raman)
                        
                        
                        
    
        
    def define_H_dipole (self):
        """Returns dipolar Hamiltonian"""
        """NOTE: I SHOULD CHANGE THE BUILD UP TO LIL_MATRIX, IT'S MUCH FASTER! """
        H_dipole = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                
                                if self.g_ij(ii,jj,a1,b1,a2,b2)!=0:
                                
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.ud_to_level['e'][a1] , self.ud_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.ud_to_level['e'][a2] , self.ud_to_level['g'][b2] , jj )
                                
                                    H_dipole = H_dipole - self.g_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i @ sigma_a2b2_j.conj().T
                                
                                    #if ii!=jj:
                                    #    print(a1,b1,a2,b2,self.g_ij(ii,jj,a1,b1,a2,b2))
                                
                                    #if (ii==jj and (a1!=a2 or b1!=b2)): print(a1,a2,b1,b2,self.g_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i @ sigma_a2b2_j.conj().T)
        
        return H_dipole
        
        
    
    
    def define_Heff_dipolar (self):
        """Saves effective Hamiltonian of dipolar interactions."""
        """NOTE: I SHOULD CHANGE THE BUILD UP TO LIL_MATRIX, IT'S MUCH FASTER! """
        self.ham_eff_dipoles = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Incoherent dipole part
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                        
                                if self.f_ij(ii,jj,a1,b1,a2,b2)!=0:        
                                #if self.f_ij(ii,ii,a1,b1,a2,b2)!=0:     # cavity      
                                
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.ud_to_level['e'][a1] , self.ud_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.ud_to_level['e'][a2] , self.ud_to_level['g'][b2] , jj )
                                
                                    self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T
                                    #self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,ii,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T    # cavity
        
        # Coherent dipole part
        self.ham_eff_dipoles = self.ham_eff_dipoles + self.define_H_dipole()   # comment out w/o cavity
        
        
        # Add extra term to break possible degeneracies of different n_e states
        if param.epsilon_ne!=0:
            for ii in range(self.Ntotal):
                for aa in range(param.deg_e):
                    sigma_aa_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['e'][aa] , ii )
                    #self.H_energies = self.H_energies + ((param.omega0+aa*param.zeeman_e)-param.omegaR) * sigma_aa_i
                    self.ham_eff_dipoles = self.ham_eff_dipoles + param.epsilon_ne * sigma_aa_i
        
        
        # Compute memory
        self.memory_sparse_Heff_dipoles = (self.ham_eff_dipoles.data.nbytes + self.ham_eff_dipoles.indptr.nbytes + self.ham_eff_dipoles.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        print("Memory for sparse Heff_dipoles: %g Gb."%(self.memory_sparse_Heff_dipoles))
        
        
        
        




####################################################################

############                EIGENSTATES                #############

####################################################################
        
        
    def compute_eigenstates (self):
        """
        Compute all eigenvalues and eigenvectors of dipole part of effective Hamiltonian
        """
        if self.memory_full_Hamiltonian<param.max_memory:
            self.evalues,self.estates = eig(self.ham_eff_dipoles.toarray())
        elif self.memory_sparse_Hamiltonian<param.max_memory:
            print("Note compute_eigenstates: Memory of full Hamiltonian too large. Using eigs, list of eigenstates incomplete.")
            self.evalues,self.estates = eigs(self.ham_eff_dipoles,k=len(self.hspace.states_list)-2,which='SR')
        else: print("ERROR compute_eigenstates: Memory of sparse Hamiltonian too large. Eigenstates not computed.")
        
        self.compute_excitations_of_eigenstates()
        
        for ii in range(len(self.evalues)):
            if self.excitations_estates[ii]>0:
                if abs(self.evalues[ii].imag)<0.0000001:
                    dark = self.estates[:,ii]
                    #print(ii,self.excitations_estates[ii],self.evalues[ii],dark)
                    
                    # Save chosen dark state to be addressed
                    #if abs(self.estates[17,ii])**2>0.0001:
                    #    self.darkstate = self.estates[:,ii]
                    #    print("\n *************")
                    #    print("This is the dark state that will be saved:\n")
                    #    print(self.darkstate)
                    #    print("\n *************")
                        
                    # Choose dark state by hand
                    # F=6, M=-4 for 5/2->7/2
                    #self.darkstate = np.zeros(len(self.evalues))
                    #self.darkstate[17] = sqrt(5/33)
                    #self.darkstate[22] = sqrt(35/66)
                    #self.darkstate[28] = sqrt(7/22)
                    
                    
                    self.darkstate = self.estates[:,ii]
                    
        #print("\n *************")
        #print("This is the dark state that will be saved:\n")
        #print(self.darkstate)
        #print("\n *************")
                    
        

        
        
        
        
    def compute_excitations_of_eigenstates (self):
        """
        WARNING: This only works for a single site right now. MODIFY!
        
        Computes to which excitation manifold each eigenstate belongs.
        If Heff is only due to dipolar interactions then each eigenstate is a superposition of states with the same number of excitations.
        If eigenstates mix different manifolds, print error.
        
        Also saves how many states are involved in each eigenstate.
        """
        if self.Ntotal==1:
            self.excitations_estates = np.full((len(self.evalues)),-1,dtype=int)
            self.nstatesinvolved_estates = np.full((len(self.evalues)),-1,dtype=int)
            for ii in range(len(self.evalues)):
                trunc_estate = np.array( [ self.truncate(abs(self.estates[jj,ii]),param.digits_trunc) for jj in range(len(self.estates[:,ii])) ] )
                
                # Save numbers of states contributing to eigenstate
                involved_states = []
                weight_involved_states = []
                for jj in range(len(trunc_estate)):
                    if trunc_estate[jj]!=0:
                        involved_states.append(jj)
                        weight_involved_states.append(trunc_estate[jj]**2)
                self.nstatesinvolved_estates[ii] = len(involved_states)
                
                # Compute occupation of involved states, check all occupations are the same, and save value of occupation.
                if len(involved_states)>0:
                    excit_involved_states = [ self.hspace.excitations_list[involved_states[jj]] for jj in range(len(involved_states)) ]
                    
                    if excit_involved_states.count(excit_involved_states[0]) == len(excit_involved_states):
                        self.excitations_estates[ii] = excit_involved_states[0]
                    else:
                        meanexcit = sum( [ excit_involved_states[jj]*weight_involved_states[jj] for jj in range(len(involved_states)) ] )
                        self.excitations_estates[ii] = round( meanexcit )
                        print(meanexcit)
                        
                        
                        #indexmax = max(range(len(weight_involved_states)), key=weight_involved_states.__getitem__)
                        #self.excitations_estates[ii] = excit_involved_states[indexmax]
                        #print(weight_involved_states[indexmax])
                        
                        #print(self.excitations_estates[ii])
                        #print(excit_involved_states)
                        #print(weight_involved_states)
                        #print(sum(weight_involved_states))
                        print("WARNING/compute_excitations_of_eigenstates: Eigenstates mix different excitation manifolds.")
                    
                else: print("ERROR/compute_excitations_of_eigenstates: Entries of eigenstate %i seem to be zero."%(ii))
                
        else: print("WARNING/compute_excitations_of_eigenstates: Can't compute excitations for more than a single site.")
        
        
        
        
    def truncate(self,number, digits) -> float:
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper
        
        
        
        
    
        
        

        
        
####################################################################

############                DYNAMICS                ###############

####################################################################
        
    def linblad_equation(self,t,psi):
        return self.linSuper @ psi
        
    def linblad_equation_timeDep(self,t,psi):
        lin_t = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        for nu in range(self.nlasers):
            temp = exp( -1j * (param.omega_laser[nu]-param.omegaR) * t ) * self.linSuper_timeDep[nu]
            lin_t = lin_t - 1j * ( temp + temp.conj().T )
        return (self.linSuper+lin_t) @ psi
        
    
    def compute_evolution_op (self):
            
        #self.evolU = expm(self.linSuper.toarray()*param.dt)
        self.evolU = sp_expm(self.linSuper*param.dt)
        
        
        
    def evolve_rho_onestep (self):
        
        if param.solver == 'exp':
            self.rhoV = self.evolU @ self.rhoV
            self.rho = self.rhoV.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
            if self.timeDep == True: print("\nWARNING: matrix exponentiation doesn't work for time-dep problem.\n")
        
        if param.solver == 'ode':
            self.solver.integrate(self.solver.t+param.dt)
            self.rho = self.solver.y.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
            self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)
        
        
    def set_solver (self,time):
        """
        Choose solver for ODE and set initial conditions.
        """
        if self.timeDep == False:
            self.solver = complex_ode(self.linblad_equation).set_integrator('dopri5')
        else:
            self.solver = complex_ode(self.linblad_equation_timeDep).set_integrator('dopri5')
        self.solver.set_initial_value(self.rhoV.reshape((len(self.rhoV))), time)
        
        #self.solver.set_initial_value( self.rho.flatten('C').reshape((self.hspace.hilbertsize**2)) , time)
        




####################################################################

#############                OUTPUT                ################

####################################################################
        
        
    def read_occup_gmanifold (self):
        """Outputs occupation of each |g,*> level, summed over all atoms."""
        out = []
        for bb in range(param.deg_g):
            observable = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            for ii in range(self.Ntotal):
                sigma_i = self.hspace.sigma_matrix( self.ud_to_level['g'][bb] , self.ud_to_level['g'][bb] , ii )
                observable = observable + sigma_i
            out.append( ( (observable @ self.rho).diagonal().sum() ).real / self.Ntotal )
        return out
        
        
        
    def read_occup_emanifold (self):
        """Outputs occupation of each |e,*> level, summed over all atoms."""
        out = []
        for aa in range(param.deg_e):
            observable = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            for ii in range(self.Ntotal):
                sigma_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['e'][aa] , ii )
                observable = observable + sigma_i
            out.append( ( (observable @ self.rho).diagonal().sum() ).real / self.Ntotal )
        return out
        
    
    
    def create_output_ops_excitedmanifold (self):
        """
        Creates operator for counting number of excitations.
        """
        self.op_excitedmanifold = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        for aa in range(param.deg_e):
            for ii in range(self.Ntotal):
                sigma_i = self.hspace.sigma_matrix( self.ud_to_level['e'][aa] , self.ud_to_level['e'][aa] , ii )
                self.op_excitedmanifold = self.op_excitedmanifold + sigma_i
        self.op_excitedmanifold = self.op_excitedmanifold / self.Ntotal 
    
        
        
    def create_output_ops_occs (self):
        """
        Creates list of occupation operators whose expectation value will be outputed.
        """
        self.output_occs_ops = []
        for nn in range(self.hspace.localhilbertsize):
            observable = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            for ii in range(self.Ntotal):
                projector_i = self.hspace.projector(nn,nn,ii)
                observable = observable + projector_i
            self.output_occs_ops.append( observable / self.Ntotal )
        
        
    def read_occs (self):
        """
        Outputs occupation of each Hilbert state, summed over all atoms.
        """
        out = []
        # Dark general F
        #out.append( (self.darkstate.conj().T @ self.rho @ self.darkstate).real )
        #if param.deg_e==2 and param.deg_g==2:
        #    out.append( (((self.rho[2,2]+self.rho[2,3]) + (self.rho[3,2]+self.rho[3,3]))/2).real )
        
        for ii in range(len(self.output_occs_ops)):
            out.append( ( (self.output_occs_ops[ii] @ self.rho).diagonal().sum() ).real )
        
        # Bright orthogonal to Dark chosen
        #bright = self.darkstate+0
        #bright[7] = self.darkstate[10]
        #bright[10] = -self.darkstate[7]
        #out.append( (bright.conj().T @ self.rho @ bright).real )
        
        #print(bright)
        
        # Bright/Dark F=1/2
        #out.append( (((self.rho[2,2]+self.rho[2,3]) + (self.rho[3,2]+self.rho[3,3]))/2).real )
        #out.append( (((self.rho[2,2]-self.rho[2,3]) - (self.rho[3,2]-self.rho[3,3]))/2).real )
        
        #out.append( ((7*(7*self.rho[2,2]+9*self.rho[2,3]) + 9*(7*self.rho[3,2]+9*self.rho[3,3]))/130).real )
        #out.append( ((9*(9*self.rho[2,2]-7*self.rho[2,3]) - 7*(9*self.rho[3,2]-7*self.rho[3,3]))/130).real )
        #out.append( self.rho[2,3].real )
        #out.append( self.rho[2,3].imag )
        return out
        
        
    def read_occs_totalF (self,eindex):
        """
        Outputs occupation of eigenstate eindex in terms of total F basis states, summed over all atoms.
        """
        out = []
        if param.deg_e==2 and param.deg_g==2 and self.Ntotal==1:
            
            ground = np.zeros(6)
            dark = self.darkstate
            bright_10 = np.zeros(6)
            bright_11 = np.zeros(6)
            bright_1m1 = np.zeros(6)
            ee = np.zeros(6)
            
            ground[0] = 1
            bright_10[2] = 1/sqrt(2)
            bright_10[3] = 1/sqrt(2)
            bright_11[4] = 1
            bright_1m1[1] = 1
            ee[5] = 1
            
            out.append( abs( ground.conj().T @ self.estates[:,eindex] )**2 )
            out.append( abs( dark.conj().T @ self.estates[:,eindex] )**2 )
            out.append( abs( bright_1m1.conj().T @ self.estates[:,eindex] )**2 )
            out.append( abs( bright_10.conj().T @ self.estates[:,eindex] )**2 )
            out.append( abs( bright_11.conj().T @ self.estates[:,eindex] )**2 )
            out.append( abs( ee.conj().T @ self.estates[:,eindex] )**2 )
            
            print(out)
            
            #print(ground)
            #print(dark)
            #print(bright_10)
            #print(bright_11)
            #print(bright_1m1)
            #print(ee)
            
            #out.append( (self.darkstate.conj().T @ self.rho @ self.darkstate).real )
            #        for ii in range(len(self.output_occs_ops)):
            #out.append( ( (self.output_occs_ops[ii] @ self.rho).diagonal().sum() ).real )
            
            
        else:
            print("\nERROR: Wrong choice of level structure or number of sites for read_occs_totalF.\n")
            
        return out
        






        













import numpy as np
from numpy import pi as PI
from numpy import sin as sin
from numpy import cos as cos

import math
from math import sqrt as sqrt


# -------------------------------------------
#           System parameters
# -------------------------------------------

"""
Units:
    Lengths in micrometers
    Frequencies in 1/microsecond
    Decay rate in 1/microsecond
"""
rescale = True     # Rescale frequencies with Gamma and lengths with lambda0

# Nature constants
c = 2.99*10**8       # Speed of light  micrometer/microsecond

### ------------------------
###     Geometry settings
### ------------------------
geometry = 'lattice'        # Choose ensemble geometry: lattice, cavity
DIM = 1     # Dimension of square lattice
N = 1       # Number of lattice sites per side in square lattice
Nx = 1      # Sites along x for rectangular array
Ny = 1      # Sites along y for rectangular array
Nz = 1      # Sites along z for rectangular array (if all ==1 then use square lattice)
latsp = 0.406   # Lattice spacing (micrometers)
onsite_prefactor = 0    # On-site pre-factor coming from integrals over wave-functions (micrometers^-3)
filling=2       # Number of atoms per site

### ---------------------
###     Atom properties
### ---------------------
deg_e=3          # Degeneracy of upper levels
deg_g=3        # Degeneracy of lower levels
deg_i=3       # Degeneracy of intermediate levels
Fe=11/2
Fg=9/2
Fi=9/2
dipole_structure='generic'         # Choose set of dipolar matrix elements:
                                    # 'two_level': dipole along quantization axis
                                    # 'Sr88': J=0 -> J=1
                                    # 'Sr87': F=9/2 -> F=7/2,9/2,11/2. Value of F given separately.
                                    # 'Yb171': F=1/2 -> F=1/2
                                    # 'generic': whatever is specified by Fg and Fe
folder_dsph='../dipole_matrix_elements_spherical/'
theta_qa=0*PI            # Spherical angles of quantization axis with respect to z.
phi_qa=0              # Spherical angles of quantization axis with respect to z.

Gamma = 0.04712                     # Decay rate of excited state (1/microsecond)
lambda0 = 0.689                     # Transition wave-length (micrometers)
k0 = 2*PI/lambda0                   # wave-vector of dipole transition (1/micrometer)
omega0 = c*k0         # frequency between levels A and B  (1/microsecond)

zeeman_g = 0*Gamma        # Constant energy shift between |g mu> and |g mu+1> (1/microsecond)
zeeman_e = 0*Gamma        # Constant energy shift between |e mu> and |e mu+1> (1/microsecond)
# 0.00089  9/2
# 0.004  11/2

epsilon_ne = 0*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that eigenvalues of different n_e are not degenerate
epsilon_ne2 = 1000*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that n_e=2 are off-resonant


### ------------------------
###     Cavity properties
### ------------------------
cavpol_z = np.array([0,0,1])     # Polarization 1 of cavity field  ---> EDIT for different q_axis?
cavpol_x = np.array([1,0,0])     # Polarization 2 of cavity field

g_inc = Gamma       # Effective atom-atom coupling (incoherent)  --> EDIT
g_coh = 0           # Effective atom-atom coupling (coherent)


### ------------------------
###     Laser properties
### ------------------------
##########   Laser g-e  ##########
levels_laser = ['ig']      # Levels connected by laser (write as 'eg', 'ig', etc). NOTE: only e,g,i. ORDER: first higher level: g < i < e < s assumed.
rabi_coupling = [2.5*Gamma]           # Proportional to electric field strength of laser and radial dipole matrix element (1/microsecond)
# PI/2*sqrt(3/2)*Gamma/(50*0.01)
detuning = [0]                  # Detuning of laser from omega0
omega_laser = [ omega0+detuning[ii] for ii in range(len(detuning)) ]             # Laser frequency (micrometers)

theta_k = [PI/2]                 # Spherical angles of k wrt axes defined by lattice
phi_k = [0]

pol_x = [1]      # Amplitude of x polarization in k-axis system
pol_y = [0]      # Amplitude of y polarization in k-axis system     e+ = -ex-iey,  e- = ex-iey


########## Raman laser properties ##########
### On resonance for now, need to derive equations with detuning
### No k-dependence for now, need to derive equations
levels_Raman = 'eis'      # Levels connected by Raman lasers (e.g. 'egs'). Last letter is intermediate state. NOTE: only e,g,i for first two letters. ORDER: first higher level: g < i < e < s assumed.
deg_s=10
Fs=9/2               # intermediate state |s,mu>
effrabi_coupling_raman = 10*Gamma   # omega_1*omega_2/delta

pol1_raman = np.array([0,0,1])      # polarization of laser 1 in fixed axis, assume k is perpendicular
pol2_raman = np.array([-1,-1j,0])/sqrt(2)      # polarization of laser 2 in fixed axis, assume k is perpendicular
#np.array([-1,-1j,0])/sqrt(2)


########## Managing the lasers ##########
switchtimes = [0,10,15]       # Time step where switching happens
switchlaser = [True,False,False]       # True: Switch lasers on, False: Switch lasers off
switchlaser_raman = [False,True,False]       # True: Switch lasers on, False: Switch lasers off

will_laser_be_used = False
will_raman_be_used = False
for ii in range(len(switchlaser)):
    if switchlaser[ii]==True: will_laser_be_used = True
for ii in range(len(switchlaser_raman)):
    if switchlaser_raman[ii]==True: will_raman_be_used = True


########### Rotating frame ##########
omegaR = omega0         # Frequency of rotating frame.
if will_raman_be_used==False and will_laser_be_used==True: omegaR = omega_laser[0]


### ------------------------
###     Initial conditions
### ------------------------
#cIC = 'test'
cIC = 'initialstate'     # Choose initial condition:
                    # 'initialstate': All atoms in state defined by 'initialstate'
                    # 'dark': superposition of |e0,g1> and |e1,g0> that is dark for Sr87 four-level 2-particles.
#initialstate = ['g0']
initialstate = ['g0','g1']
#initialstate = ['e2','e1','e0']
#initialstate = ['g0','g3','g2','g3']
#initialstate = ['e4','e3','e2','e1','e0']
#initialstate = ['e5','e4','e3','e2','e1','e0']
#initialstate = ['e6','e5','e4','e3','e2','e1','e0']

### ------------------------
###         Other
### ------------------------
digits_trunc = 6        # Number of digits to truncate when computing excitation manifold of eigenstates
      
### ------------------------
###         Output
### ------------------------
#outfolder = './data'
outfolder = '.'
#outfolder = './data/3P1_preparation'
output_occupations = True
output_eigenstates = False
output_stateslist = True
append = ''
#append = '_zg%g_ze%g_effome%g_ramsey_symZee'%(zeeman_g/Gamma,zeeman_e/Gamma,effrabi_coupling_raman/Gamma)
#append = '_Ns%i_effrabi%g'%(deg_s,effrabi_coupling_raman/Gamma)
#append = '_ze%g_ome%g'%(zeeman_e/Gamma,rabi_coupling[0]/Gamma)
#append = '_ze%g_ome%g'%(zeeman_e/Gamma,rabi_coupling[0]/Gamma)
#append = '_ene%g'%(epsilon_ne/Gamma)
#append = '_thetaq%g'%(theta_qa/PI)
#append = '_ze%g'%(zeeman_e/Gamma)

### ------------------------
###         Solver
### ------------------------
#solver = 'exp'         # Solve dynamics by exponentiating Linblad
solver = 'ode'          # Solve system of differential equations


### ------------------------
###     Time evolution
### ------------------------
dt=0.05/Gamma
Nt=100
max_memory=1        # Maximal memory allowed for superoperator. If more needed, abort program.





###
### RESCALE PARAMETERS WITH GAMMA AND LAMBDA0
###

"""
Only dimensionless quantities:

Frequency = Frequency / Gamma
Length = Length / lambda0

Note that omega = c*k ---->  omega = c/(lambda0*Gamma) k , where omega and k are rescaled with Gamma and lambda0
"""

if rescale == True:
    
    c = c/(lambda0*Gamma)   # Frequency*Length
    
    latsp = latsp/lambda0   # Length
    #onsite_prefactor = onsite_prefactor*lambda0**3    # 1/Length**3  ## This is dimensionless now
    k0 = k0*lambda0                   # 1/Length
    
    omega0 = omega0/Gamma            # Frequency
    zeeman_g = zeeman_g/Gamma        # Frequency
    zeeman_e = zeeman_e/Gamma        # Frequency
    epsilon_ne = epsilon_ne/Gamma       # Frequency
    
    rabi_coupling = [ rabi_coupling[nu]/Gamma for nu in range(len(rabi_coupling)) ]   # Frequency
    detuning = [ detuning[nu]/Gamma for nu in range(len(detuning)) ]                  # Frequency
    omega_laser = [ omega_laser[nu]/Gamma for nu in range(len(omega_laser)) ]         # Frequency
    
    effrabi_coupling_raman = effrabi_coupling_raman/Gamma
    
    omegaR = omegaR/Gamma         # Frequency
    
    dt=dt*Gamma
    
    g_inc = g_inc/Gamma
    g_coh = g_coh/Gamma
    
    Gamma=1
    lambda0=1




###
### CHECKS
###

if deg_g>2*Fg+1 or deg_e>2*Fe+1 or deg_i>2*Fi+1 or deg_s>2*Fs+1:
    print('\nERROR/parameters: number of levels larger than degeneracy of F.\n')

check_numberL = [len(levels_laser),len(rabi_coupling),len(detuning),len(theta_k),len(phi_k),len(pol_x),len(pol_y)]
if check_numberL.count(check_numberL[0]) != len(check_numberL):
    print('\nERROR/parameters: Length of laser parameter arrays inconsistent.\n')
    

check_numberSw = [len(switchtimes),len(switchlaser)]
if check_numberSw.count(check_numberSw[0]) != len(check_numberSw):
    print('\nERROR/parameters: Length of switch parameter arrays inconsistent.\n')


if cIC == 'initialstate':
    if len(initialstate) != filling:
        print('\nERROR/parameters: Initial state specified incorrect length for chosen filling.\n')
        
        
if epsilon_ne!=0:
    print('\nINFO/parameters: bias term epsilon_ne is nonzero.\n')
    
    
if deg_i>0:
    print('\nINFO/parameters: deg_i is nonzero.\n')








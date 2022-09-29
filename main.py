# 

import numpy as np
import matplotlib.pyplot as plt
#import cmath
#from cmath import exp as exp
from numpy import pi as PI
from numpy import exp as exp
from numpy import sin as sin
from numpy import cos as cos

from scipy.optimize import curve_fit

import time
import sys


import parameters as param
import dipolar_system


"""
Explain code here


"""



# -------------------------------------------
#           Functions
# -------------------------------------------
        
        
# Prints execution time in hours, minutes and seconds
def print_time(t):
    hours = t//3600;
    minutes = (t-3600*hours)//60;
    seconds = (t-3600*hours-60*minutes);
    print("Execution time: %i hours, %i minutes, %g seconds"%(hours,minutes,seconds));






# -------------------------------------------
#           Set up system
# -------------------------------------------

starttime = time.time()
lasttime = starttime

print("\n-------> Setting up system\n")

dipsys = dipolar_system.Dipolar_System()

if param.output_eigenstates:
    if dipsys.memory_full_Hamiltonian>param.max_memory:
        print("Memory of full Hamiltonian is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
        sys.exit()

if param.Nt>0:
    if dipsys.memory_estimate_sparse_Linblad>param.max_memory:
        print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
        sys.exit()



# -------------------------------------------
#           Compute eigenvectors
# -------------------------------------------

print("\n-------> Computing eigenvectors\n")

if param.output_eigenstates:
    
    dipsys.define_Heff_dipolar()
    
    dipsys.compute_eigenstates()


    print("\nTime computing eigenvectors.")
    print_time(time.time()-lasttime)
    lasttime = time.time()
    
else: print("No.")



# -------------------------------------------
#                 Dynamics
# -------------------------------------------

print("\n-------> Computing dynamics\n")

if param.Nt>0:

    # --------
    ##########  Set up initial time
    # --------

    phase = 0       # Traces stage of evolution, if Hamiltonian has sudden changes in time, e.g. laser switch on/off

    dipsys.decide_if_timeDep(phase)

    dipsys.choose_initial_condition()

    dipsys.save_partial_HamLin() # Computes ham_eff_dipoles again. Change code to avoid it.

    dipsys.define_linblad_superop(phase)

    dipsys.compute_memory()
    if dipsys.memory_sparse_Linblad+dipsys.memory_sparse_Hamiltonian > param.max_memory:
        print("Memory is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
        sys.exit()

    if param.solver == 'exp': dipsys.compute_evolution_op()
    if param.solver == 'ode': dipsys.set_solver(0)


    print("\nTime for constructing Linblad at t=0.")
    print_time(time.time()-lasttime)
    lasttime = time.time()


    # --------
    ##########  Evolve
    # --------

    times=[ 0 ]

    # Prepare output
    if param.output_occupations:
        dipsys.create_output_ops_occs()
        out_occup_states = [ dipsys.read_occs() ]

    # Time evolution
    for tt in range(1,param.Nt+1):
    
        print(times[tt-1])
    
        if param.solver == 'ode':
            if dipsys.solver.successful():
                dipsys.evolve_rho_onestep()
            else: print("\nERROR: Problem with solver, returns unsuccessful.\n")
            times.append(dipsys.solver.t)
    
        if param.solver == 'exp':
            dipsys.evolve_rho_onestep()
            times.append( tt*param.dt )
    
        if param.output_occupations: out_occup_states.append(dipsys.read_occs())    
    
        if len(param.switchtimes)>phase+1:
            if param.switchtimes[phase+1]==tt:
                phase = phase + 1
                dipsys.decide_if_timeDep(phase)
                dipsys.define_linblad_superop(phase)
                if param.solver == 'exp': dipsys.compute_evolution_op()
                if param.solver == 'ode': dipsys.set_solver(dipsys.solver.t)
            

    #print(dipsys.rho)
    print("Purity:")
    print(np.trace(dipsys.rho@dipsys.rho))

    print("\nTime for evolving.")
    print_time(time.time()-lasttime)
    lasttime = time.time()
    
    
else: print("No.")



# -------------------------------------------
#           Output data
# -------------------------------------------

#file_states_occ = open('%s/states_occ_%s_fill%i_IC-%s_selfI%g.pdf'%(param.outfolder, param.dipole_structure, param.filling, param.cIC, param.onsite_prefactor),'w')

comments = ''
comments = comments + '# geometry = %s\n'%(param.geometry)
comments = comments + '# N = %i\n'%(dipsys.Ntotal)
comments = comments + '# filling = %i\n'%(param.filling)
comments = comments + '# latsp = %g\n'%(param.latsp)
comments = comments + '# onsite_prefactor = %g\n'%(param.onsite_prefactor)
comments = comments + '# \n'

comments = comments + '# deg_e = %i\n'%(param.deg_e)
comments = comments + '# deg_g = %i\n'%(param.deg_g)
comments = comments + '# deg_i = %i\n'%(param.deg_i)
comments = comments + '# Fe = %g\n'%(param.Fe)
comments = comments + '# Fg = %g\n'%(param.Fg)
comments = comments + '# Fi = %g\n'%(param.Fi)
comments = comments + '# dipole_structure = %s\n'%(param.dipole_structure)
comments = comments + '# \n'

comments = comments + '# theta_qa = %g\n'%(param.theta_qa)
comments = comments + '# phi_qa = %g\n'%(param.phi_qa)
comments = comments + '# Gamma = %g\n'%(param.Gamma)
comments = comments + '# lambda0 = %g\n'%(param.lambda0)
comments = comments + '# zeeman_g = %g\n'%(param.zeeman_g)
comments = comments + '# zeeman_e = %g\n'%(param.zeeman_e)
comments = comments + '# epsilon_ne = %g\n'%(param.epsilon_ne)
comments = comments + '# epsilon_ne2 = %g\n'%(param.epsilon_ne2)
comments = comments + '# \n'

comments = comments + '# Number of lasers = %i\n'%(len(param.rabi_coupling))
comments = comments + '# levels_laser = %s\n'%( ', '.join(param.levels_laser) )
comments = comments + '# rabi_coupling = %s\n'%( ', '.join([str(param.rabi_coupling[ii]) for ii in range(len(param.rabi_coupling))]) )
comments = comments + '# detuning = %s\n'%( ', '.join([str(param.detuning[ii]) for ii in range(len(param.detuning))]) )
comments = comments + '# theta_k = %s\n'%( ', '.join([str(param.theta_k[ii]) for ii in range(len(param.theta_k))]) )
comments = comments + '# phi_k = %s\n'%( ', '.join([str(param.phi_k[ii]) for ii in range(len(param.phi_k))]) )
comments = comments + '# pol_x = %s\n'%( ', '.join([str(param.pol_x[ii]) for ii in range(len(param.pol_x))]) )
comments = comments + '# pol_y = %s\n'%( ', '.join([str(param.pol_y[ii]) for ii in range(len(param.pol_y))]) )
comments = comments + '# omegaR = %g\n'%(param.omegaR)
comments = comments + '# \n'

comments = comments + '# Number of Raman lasers = 1\n'
comments = comments + '# levels_laser = %s\n'%( ', '.join(param.levels_Raman) )
comments = comments + '# deg_s = %i\n'%(param.deg_s)
comments = comments + '# Fs = %g\n'%(param.Fs)
comments = comments + '# rabi_coupling = %s\n'%(param.effrabi_coupling_raman)
comments = comments + '# pol_1_raman = %s\n'%( '('+ ', '.join([str(param.pol1_raman[ii]) for ii in range(len(param.pol1_raman))]) +')')
comments = comments + '# pol_2_raman = %s\n'%( '('+ ', '.join([str(param.pol2_raman[ii]) for ii in range(len(param.pol2_raman))]) +')')
comments = comments + '# \n'

comments = comments + '# Laser switch times = %s\n'%( ', '.join([str(param.switchtimes[ii]) for ii in range(len(param.switchtimes))]) )
comments = comments + '# Laser on/off = %s\n'%( ', '.join([str(param.switchlaser[ii]) for ii in range(len(param.switchlaser))]) )
comments = comments + '# Raman on/off = %s\n'%( ', '.join([str(param.switchlaser_raman[ii]) for ii in range(len(param.switchlaser_raman))]) )
comments = comments + '# \n'

comments = comments + '# IC = %s\n'%(param.cIC)
comments = comments + '# initialstate = %s\n'%('| '+' '.join(param.initialstate)+' >')
comments = comments + '# solver = %s\n'%(param.solver)
comments = comments + '# \n# \n'





# States occupations
if param.output_occupations:
    filename = '%s/states_occ_%s_fill%i_Ng%i_Ne%i_Ni%i_C%g_IC%s%s.txt'%(param.outfolder, param.dipole_structure, param.filling,\
                 param.deg_g, param.deg_e, param.deg_i, param.onsite_prefactor, ''.join(param.initialstate), param.append)
    with open(filename,'w') as f:
        f.write(comments)
        if param.deg_e==2 and param.deg_g==2:
            f.write('# Col 1: time | Col 2: dark state | Col 3: superrad state | Col >= 4: remaining states.\n')
        else:
            f.write('# Col 1: time | Col 2: dark state | Col > 3: remaining states.\n')
        f.write('# \n# \n')
        output_data = [ [ times[tt] ] + out_occup_states[tt] for tt in range(len(times)) ]
        np.savetxt(f,output_data,fmt='%.6g')
    

# Eigenstates
if param.output_eigenstates:
    filename = '%s/eigenstates_%s_fill%i_Ng%i_Ne%i_Ni%i_C%g%s.txt'%(param.outfolder, param.dipole_structure, param.filling,\
                 param.deg_g, param.deg_e, param.deg_i, param.onsite_prefactor, param.append)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: energy | Col 2: decay rate | Col 3: number of excitations | Col 4: number of states involved | Col >=5: square_abs of amplitudes of eigenstate.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*2 + ['%i\t%i'] + [floatfmt]*dipsys.hspace.hilbertsize )
        output_data = [ [ dipsys.evalues[ii].real , -2*dipsys.evalues[ii].imag , dipsys.excitations_estates[ii] , dipsys.nstatesinvolved_estates[ii] ] + list(abs(dipsys.estates[:,ii])**2) for ii in range(len(dipsys.evalues)) ]
        #formatstring = '\t'.join( [floatfmt]*2 + [floatfmt]*dipsys.hspace.hilbertsize )
        #output_data = [ [ dipsys.evalues[ii].real , -2*dipsys.evalues[ii].imag ] + list(abs(dipsys.estates[:,ii])**2) for ii in range(len(dipsys.evalues)) ]
        np.savetxt(f,output_data,fmt=formatstring)
        
        
    """ Occupations in EIGENSTATE BASIS FOR 1/2-1/2
    filename = '%s/eigenstatesTotalF_%s_fill%i_Ng%i_Ne%i_C%g%s.txt'%(param.outfolder, param.dipole_structure, param.filling,\
                 param.deg_g, param.deg_e, param.onsite_prefactor, param.append)
    if param.deg_e==2 and param.deg_g==2:
        with open(filename,'w') as f:
            f.write(comments)
            f.write('# Col 1: energy | Col 2: decay rate | Col 3: number of excitations | Col 4: number of states involved |\n# Col >=5: square_abs of amplitudes of eigenstate in total F basis. 5: ground, 6: dark, 7: |1,M=-1>, 8: |1,M=0>, 9: |1,M=1>, 10: ee.\n')
            f.write('# \n# \n')
            floatfmt = '%.6g'
            formatstring = '\t'.join( [floatfmt]*2 + ['%i\t%i'] + [floatfmt]*dipsys.hspace.hilbertsize )
            output_data = [ [ dipsys.evalues[ii].real , -2*dipsys.evalues[ii].imag , dipsys.excitations_estates[ii] , dipsys.nstatesinvolved_estates[ii] ] + dipsys.read_occs_totalF(ii) for ii in range(len(dipsys.evalues)) ]
            np.savetxt(f,output_data,fmt=formatstring)
    """
        
    #dipsys.read_occs_totalF(5)

    #for ii in range(len(dipsys.estates[:,6])):
        
        #print(ii, dipsys.truncate(dipsys.estates[ii,6].real,5), dipsys.truncate(dipsys.estates[ii,6].imag,5) )
        #print(dipsys.truncate(4.12452151,4))


print("\nTime for file output.")
print_time(time.time()-lasttime)
lasttime = time.time()




# -------------------------------------------
#           Plots
# -------------------------------------------



###
### Plot individual occupancies of each state.
###
#for nn in range(dipsys.hspace.localhilbertsize):
if param.output_occupations:
    for nn in range(len(out_occup_states[0])):
        plt.plot( times, [out_occup_states[ii][nn] for ii in range(len(out_occup_states))] , label=r'$| %i \rangle \langle %i |$'%(nn,nn))
    #plt.plot( times, exp(-np.array(times)) , label='exp')
    plt.xlabel('Time: ' + r'$t$')
    plt.ylabel('State occupancies: ' + r'$| \alpha \rangle \langle \alpha |$')
    plt.xlim(0,param.Nt*param.dt)
    plt.legend(loc='upper right')
    plt.savefig('%s/states_occ_%s_fill%i_Ng%i_Ne%i_C%g_IC%s%s.pdf'%(param.outfolder, param.dipole_structure, param.filling,\
                     param.deg_g, param.deg_e, param.onsite_prefactor, ''.join(param.initialstate), param.append))

"""

def fitfunc(x,amp,gamma,offset):
    return amp*np.exp(-gamma*x) + offset
    
def shorttime(t,offset,slope):
    return offset+slope*t
    
def shorttime2(t,offset,slope,curv):
    return offset+slope*t+curv*t**2
    
popt, pcov = curve_fit(fitfunc, times, out_occup_totalup)








###
### Plot individual occupancies of each level.
###
plt.figure()
for aa in range(param.deg_e):
    plt.plot( times, [out_occup_up[ii][aa] for ii in range(len(out_occup_up))] , label=r'$| up,%i \rangle \langle up,%i |$'%(aa,aa))
for bb in range(param.deg_g):
    plt.plot( times, [out_occup_down[ii][bb] for ii in range(len(out_occup_down))] , label=r'$| down,%i \rangle \langle down,%i |$'%(bb,bb) )
plt.xlabel('Time: ' + r'$t$')
plt.ylabel('Level occupancies: ' + r'$| \alpha \rangle \langle \alpha |$')
plt.xlim(0,param.Nt*param.dt)
plt.legend(loc='upper right')
plt.savefig('%s/level_occ_%s_fill%i_IC-%s_selfI%g.pdf'%(param.outfolder, param.dipole_structure, param.filling, param.cIC, param.onsite_prefactor))



###
### Plot total occupancies of up and down.
###
plt.figure()
plt.plot( times, out_occup_totalup , label=r'$| up \rangle \langle up |$')
plt.plot( times, out_occup_totaldown , label=r'$| down \rangle \langle down |$')
plt.plot( times, fitfunc(np.array(times),*popt) , 'r--', label=r'Fit: A=%.2g, $\gamma$=%.2g, $b=%.2g$'%tuple(popt)+'\n'+r'Single: $\Gamma$=%g'%(param.Gamma))
#plt.plot( times[0:10], shorttime(np.array(times[0:10]),1,-param.Gamma*49/99) , 'g--', label='short-time')
#plt.plot( times[0:10], shorttime2(np.array(times[0:10]),1,-param.Gamma*49/99,0.5*param.Gamma**2*((49/99)**2+(7/11)**2)) , 'b--', label='short-time')
plt.xlabel('Time: ' + r'$t$')
plt.ylabel('Total occupancies: ' + r'$| \alpha \rangle \langle \alpha |$')
plt.xlim(0,param.Nt*param.dt)
plt.legend(loc='upper right')
plt.savefig('%s/total_occ_%s_fill%i_IC-%s_selfI%g.pdf'%(param.outfolder, param.dipole_structure, param.filling, param.cIC, param.onsite_prefactor))


print(out_occup_states[-1])


#print(dipsys.g_ij(0,0,1,1,1,1))

"""
                     
print("\nTime for plotting.")
print_time(time.time()-lasttime)
lasttime = time.time()



print("\nTotal time.")
print_time(time.time()-starttime)




"""
    Lattice Test
"""

#tstlattice = lattice.Lattice()
#print(tstlattice.dim)
#print(tstlattice.Nlist)
#print(tstlattice.Ntotal)

#inds=[]

#for x in range(0,tstlattice.Nlist[0]):
#    for y in range(0,tstlattice.Nlist[1]):
#        inds.append([x,y])
        
#poss = list(map(tstlattice.get_array_position , inds))

#poss = [tstlattice.get_array_position(item) for item in inds]
#backinds = [tstlattice.get_indices(item) for item in poss]

#print( inds)
#print(poss)
#print( backinds )






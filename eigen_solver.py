"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.io
import scipy.io as sio
from scipy.integrate import quad
import dedalus.public as de
#from mpi4py import MPI
#CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

# Global parameters
Nr = 500
kx_global = np.linspace(2., 2., 1)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', Nr, interval=(-1., 1.))
domain = de.Domain([z_basis], grid_dtype=np.complex128) #, comm=MPI.COMM_SELF)

#r_basis = de.Chebyshev('r', Nr, interval=(0., 4.)) #, dealias=3/2)
#domain = de.Domain([r_basis], grid_dtype=np.complex128) #, comm=MPI.COMM_SELF)

# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
problem = de.EVP(domain, variables=['u', 'w', 'h', 'uz', 'wz', 'hz'], eigenvalue='omega', tolerance=1e-16)
#problem = de.EVP(domain, variables=['u', 'w', 'h', 'uz', 'hz'], eigenvalue='omega', tolerance=1e-10)
problem.meta[:]['z']['dirichlet'] = True

#problem = de.EVP(domain, variables=['u', 'w', 'h','ur', 'wr', 'hr'], eigenvalue='omega')
#problem.meta[:]['r']['dirichlet'] = True

problem.parameters['kx'] = 1
problem.parameters['L'] = L = 1.
problem.parameters['nu'] = 1.e-6
# +ve epsilon -> cyclonic, and -ve epsilon -> anticyclonic vortex
problem.parameters['epsilon'] = epsilon = +0.1414
problem.parameters['alpha'] = alpha = 4.

z = domain.grid(0)

#r = domain.grid(0)

r0 = domain.new_field()
V = domain.new_field()
Vr = domain.new_field()
H = domain.new_field()
Hr = domain.new_field()

r0['g'] = L*(1.+z)/(1.-z)
problem.parameters['r0'] = r0 

V['g'] = epsilon*r0['g']**(0.5*alpha)*np.exp( 0.5*(-r0['g']**alpha+1.) )
problem.parameters['V'] = V

Vr['g'] = epsilon*alpha/(2.*r0['g'])*r0['g']**(0.5*alpha)*np.exp( 0.5*(-r0['g']**alpha+1.) )*( 1.-r0['g']**alpha )
problem.parameters['Vr'] = Vr

dr = domain.grid_spacing(0)

Hr['g'] = (V['g']/r0['g'] + 1.)*V['g']
problem.parameters['Hr'] = Hr

H['g'] = integrate.cumtrapz( Hr['g'], r0['g'], initial=0. )
max_ = np.max(H['g'])
H['g'] = 1. + H['g'] - max_
problem.parameters['H'] = H

#sio.savemat( 'z.mat', {'z':z} )
sio.savemat( 'reig.mat', {'reig':r0['g']} )
#sio.savemat( 'h.mat', {'h0':H['g']} )
#sio.savemat( 'v.mat', {'v':V['g']} )
#sio.savemat( 'vr.mat', {'vr':Vr['g']} )
#sio.savemat( 'hr.mat', {'hr':Hr['g']} )

problem.substitutions['dx(A)'] = "1j*kx*A"
problem.substitutions['dt(A)'] = "-1j*omega*A"

problem.substitutions['Lap(A)'] = "- A + dx(dx(A))"
problem.substitutions['Lap_r(A)'] = "- 2*dx(A)"
problem.substitutions['Lap_x(A)'] = "+ 2*dx(A)"

problem.substitutions['c1'] = "L**2*(1+z)**2"
problem.substitutions['c2'] = "L*(1-z**2)"

problem.substitutions['c3'] = "(1-z**2)**2*(1-z)**2"
problem.substitutions['c4'] = "(1-z**2)**2*(1-z)"
problem.substitutions['c5'] = "(1-z**2)*(1-z)**2"


problem.add_equation("c1*dt(u) + c2*V*dx(u) - (c1+2*V*c2)*w + 0.5*c2*c2/L*hz - nu*0.25*c3*dz(uz) + nu*0.5*c4*uz - nu*0.5*c5*dz(uz) - nu*(1-z)**2*( Lap(u) + Lap_r(w) ) = 0 ")
problem.add_equation("c1*dt(w) + c2*V*dx(w) + (c1*Vr+c1+V*c2)*u + c2*dx(h)   - nu*0.25*c3*dz(wz) + nu*0.5*c4*wz - nu*0.5*c5*dz(wz) - nu*(1-z)**2*( Lap(w) + Lap_x(u) ) = 0")
problem.add_equation("(1+z)*dt(h) + V/L*(1-z)*dx(h) + 0.5*H/L*(1-z)**2*(1+z)*uz + Hr*(1+z)*u + H/L*(1-z)*u + H/L*(1-z)*dx(w) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("hz - dz(h) = 0")


problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
#problem.add_bc("left(h) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0")
problem.add_bc("right(h) = 0")

solver = problem.build_solver()

# Create function to compute max growth rate for given kx
def max_growth_rate(kx):
	logger.info('Computing max growth rate for kx = %f' %kx)
	# Change kx parameter
	problem.namespace['kx'].value = kx
	#solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
	# Solve for eigenvalues with sparse search near zero, rebuilding NCCs
	solver.solve_sparse(solver.pencils[0], N=25, target=0.1, rebuild_coeffs=True)
	
	return solver
	
	
# Compute growth rate over local wavenumbers
kx_local = kx_global #[CW.rank::CW.size]
t1 = time.time()
solver = max_growth_rate(kx_local)
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Filter infinite/nan eigenmodes
finite = np.isfinite(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[finite]
solver.eigenvectors = solver.eigenvectors[:,finite]

#print('growth rate = ', solver.eigenvalues)

# Sort eigenmodes by eigenvalue
solver.eigenvalues = solver.eigenvalues[ np.abs(solver.eigenvalues)< 1. ]
order = np.argsort(-1.*solver.eigenvalues.imag)
solver.eigenvalues = solver.eigenvalues[order]
solver.eigenvectors = solver.eigenvectors[:, order]

print('shape = ', np.shape(solver.eigenvectors))

pos_eigs = np.where(solver.eigenvalues[:]>0)[0]
solver.set_state(pos_eigs[0])

growth = solver.eigenvalues.imag
phase  = solver.eigenvalues.real

print( 'growth rate = ', growth[0] )
print( 'phase speed = ', phase[0] )

u = solver.state['u']
w = solver.state['w']
h = solver.state['h']

#u['g'] = u['g']/1j
#h['g'] = h['g']/1j 

phase = np.arctan(u['g'].imag/u['g'].real)
phase = np.sum(phase*np.abs(u['g'])) / np.sum(np.abs(u['g']))
u['g'] *= np.exp(-1j*phase)

phase = np.arctan(w['g'].imag/w['g'].real)
phase = np.sum(phase*np.abs(w['g'])) / np.sum(np.abs(w['g']))
w['g'] *= np.exp(-1j*phase)

phase = np.arctan(h['g'].imag/h['g'].real)
phase = np.sum(phase*np.abs(h['g'])) / np.sum(np.abs(h['g']))
h['g'] *= np.exp(-1j*phase)

#h['g'] /= np.max(np.abs(h['g']))
#w['g'] /= np.max(np.abs(w['g']))
#u['g'] /= np.max(np.abs(u['g']))

#sio.savemat( 'v_pertub.mat', {'vp':w['g']} )

fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot(311)
ax1.set_title(r'$u$')
ax2 = plt.subplot(312)
ax2.set_title(r'$v$')
ax3 = plt.subplot(313)
ax3.set_title(r'$\eta$')

y_basis1 = de.Chebyshev('y', Nr, interval=(-1, 1))
domain1 = de.Domain([y_basis1], grid_dtype=np.complex128)
Y1 = domain1.grid(0, scales=domain.dealias)


print( 'max-u = ', np.max(u['g'].imag), 'min-u = ', np.min(u['g'].imag) )
print( 'max-w = ', np.max(w['g'].imag), 'min-w = ', np.min(w['g'].imag) )
print( 'max-h = ', np.max(h['g'].imag), 'min-h = ', np.min(h['g'].imag) )

#n1 = -1
ax1.plot(r0['g'], u['g'].real, color='C0', label='N={}'.format(Nr))
ax1.plot(r0['g'], u['g'].imag, '--', color='C0', label='N={}'.format(Nr))

ax2.plot(r0['g'], w['g'].real, color='C0', label='N={}'.format(Nr))
ax2.plot(r0['g'], w['g'].imag, '--', color='C0', label='N={}'.format(Nr))

ax3.plot(r0['g'], h['g'].real, color='C0', label='N={}'.format(Nr))
ax3.plot(r0['g'], h['g'].imag, '--', color='C0', label='N={}'.format(Nr))

AX = [ax1, ax2, ax3]
for ax in AX:
	ax.set_xlim([0, 6.])
	ax.axhline(y=0, linewidth=0.7, color='k', linestyle='--')
	plt.savefig('eigen_function%s.png' %Nr)
	

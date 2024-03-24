import booz_xform as bx
from simsopt.mhd.vmec import Vmec
import argparse
import os
from simsopt._core.optimizable import load, save
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field import BiotSavart
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo import CurveLength
import pickle
from pystellplot.VMEC.plot_surfaces import plot_vmec_surfaces

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path", dest="path", default=None)
parser.add_argument("--show", dest="show", default=False, action="store_true")

# Prepare args
args = parser.parse_args()

this_path = os.path.join(os.getcwd(), args.path)
figure_path = os.path.join( this_path, 'figure')
os.makedirs(figure_path, exist_ok=True)

os.chdir(this_path)



with open(os.path.join(this_path, 'outputs.pckl'),'rb') as f:
    out = pickle.load(f)

# Plot target function values
fig, axs = plt.subplots(2, 4, figsize=(16,9))
axs[0,0].semilogy(np.array(out['J']) / out['J'][0], label=r'$J$')
try:
    axs[0,0].semilogy(np.array(out['Jplasma']) / out['Jplasma'][0], label=r'$J_{plasma}$')
except ValueError as e:
    print('Jplasma is corrupted')
try:
    axs[0,0].semilogy(np.array(out['Jcoils']) / out['Jcoils'][0], label=r'$J_{coils}$')
except ValueError as e:
    print('Jcoils is corrupted')
axs[0,0].set_xlabel('Fct evaluation')
axs[0,0].set_ylabel('Normalized target')
axs[0,0].legend()

axs[0,1].plot(out['iota_axis'], label=r'$\iota_{axis}$')
axs[0,1].plot(out['iota_edge'], label=r'$\iota_{edge}$')
axs[0,1].plot(out['mean_iota'], label=r'$\iota_{mean}$')
axs[0,1].set_xlabel('Fct evaluation')
axs[0,1].set_ylabel('Rotational transform')
axs[0,1].legend()

axs[0,2].plot(out['aspect'])
axs[0,2].set_xlabel('Fct evaluation')
axs[0,2].set_ylabel('Aspect ratio')

axs[1,0].semilogy(out['QuadFlux'])
axs[1,0].set_xlabel('Fct evaluation')
axs[1,0].set_ylabel('Quadratic flux')

axs[1,1].plot(out['min_CS'], label='Coil-Surface')
axs[1,1].plot(out['min_CC'], label='Coil-Coil')
axs[1,1].legend()
axs[1,1].set_xlabel('Fct evaluation')
axs[1,1].set_ylabel('Minimum distance')

axs[1,2].plot(out['IL_length'], label='IL length')
axs[1,2].plot(out['IL_msc'], label='IL mean square curvature')
axs[1,2].plot(out['IL_max_curvature'], label='IL max curvature')
axs[1,2].legend()
axs[1,2].set_xlabel('Fct evaluation')
axs[1,2].set_ylabel('IL coil metric')

axs[1,3].semilogy(out['vmec']['fsqr'], label='fsqr')
axs[1,3].semilogy(out['vmec']['fsqz'], label='fsqz')
axs[1,3].semilogy(out['vmec']['fsql'], label='fsql')
axs[1,3].legend()
axs[1,2].set_xlabel('Fct evaluation')
axs[1,2].set_ylabel('VMEC convergence metric')

plt.savefig(os.path.join(figure_path, 'metric_evolution'))
plt.tight_layout()
if args.show:
    plt.show()







v = Vmec('input.final')
bs = load(os.path.join(this_path, 'coils/bs_output.json'))
v.run()

# Run boozer
b = bx.Booz_xform()
b.read_wout(v.output_file)
b.mboz = 48
b.nboz = 48
b.run()
b.write_boozmn("boozmn.nc")
s = b.s_b

# Evaluate figures of merit
bmnc = b.bmnc_b
xm = b.xm_b
xn = b.xn_b
f = np.sqrt(np.sum(bmnc[xn!=0,:]**2,axis=0)/np.sum(bmnc**2,axis=0))
fqs = np.mean(f)
mean_iota = np.mean(b.iota)
volume = v.boundary.volume()
major_radius = v.boundary.major_radius()
minor_radius = v.boundary.minor_radius()
aspect_ratio = v.boundary.aspect_ratio()
il_current = np.abs(bs.coils[0].current.get_value())
pf_current = np.abs(bs.coils[2].current.get_value())
ll = CurveLength( bs.coils[0].curve )
il_length = ll.J()

with open( os.path.join(this_path, 'figures_of_merit.txt'), 'w' ) as file:
          file.write(f'Mean iota = {mean_iota:.2E}\n')
          file.write(f'f_quasisymmetry = {fqs:.2E}\n')
          file.write(f'Volume = {volume:.2E}\n')
          file.write(f'Major radius = {major_radius:.2E}\n')
          file.write(f'Minor radius = {minor_radius:.2E}\n')
          file.write(f'Aspect ratio = {aspect_ratio:.2E}\n')
          file.write(f'IL coil current = {il_current:.2E}\n')
          file.write(f'PF coil current = {pf_current:.2E}\n')
          file.write(f'IL coil length = {il_length:.2E}\n')

# Plot quaissymmetry
plt.figure()
plt.plot(s,f)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('f [quasisymmetry metric]')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'fquasisymmetry'))
if args.show:
    plt.show()

# Plot rotational transform
plt.figure()
plt.plot(s,b.iota)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('iota')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'iota'))
if args.show:
    plt.show()

# Plot |B| contours
plt.figure()
bx.surfplot(b, js=-1)
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'modB.png'))
if args.show:
    plt.show()

# plot magnetic well
plt.figure()
plt.plot(v.s_half_grid,v.wout.vp[1::])
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('Vp [radial derivative of volume]')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'magnetic_well'))
if args.show:
    plt.show()

# Plot B.n
surf = v.boundary
theta = surf.quadpoints_theta
phi = surf.quadpoints_phi
ntheta = theta.size
nphi = phi.size
bs.set_points(surf.gamma().reshape((-1,3)))
Bdotn = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
modB = bs.AbsB().reshape((nphi,ntheta))

fig, ax = plt.subplots()
c = ax.contourf(theta,phi,Bdotn / modB)
plt.colorbar(c)
ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$\theta$')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, 'normal_field_error.png'))

# Run and plot Poincare section
bs = load(os.path.join(this_path, 'coils/bs_output.json'))
vmec_surf = SurfaceRZFourier.from_wout(v.output_file) # vmec surface
vmec_surf_1 = SurfaceRZFourier.from_wout(v.output_file) # Expanded vmec surface
vmec_surf_1.extend_via_normal(0.1)
nfp = vmec_surf.nfp

Rmaj = vmec_surf.major_radius()
r0 = vmec_surf.minor_radius()
sc_fieldline = SurfaceClassifier(vmec_surf_1, h=0.01, p=3)
nfieldlines = 50
tmax_fl = 2500
degree = 4

def trace_fieldlines(bfield,label):
    # Set up initial conditions - 
    R0 = np.linspace(Rmaj-2*r0, Rmaj+2*r0, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-8,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(figure_path, 'poincare'), dpi=150,surf=vmec_surf,mark_lost=True)
    return fieldlines_phi_hits

hits = trace_fieldlines(bs, 'vmec')


# Plot without WPs
bs = load(os.path.join(this_path, 'coils/bs_output.json'))
coils = bs.coils
if len(coils)>4:
    coils = coils[:4]
    bs = BiotSavart( coils )

    surf = v.boundary
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    ntheta = theta.size
    nphi = phi.size
    bs.set_points(surf.gamma().reshape((-1,3)))
    Bdotn = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
    modB = bs.AbsB().reshape((nphi,ntheta))
    
    fig, ax = plt.subplots(figsize=(12,5))
    c = ax.contourf(theta,phi,Bdotn / modB)
    plt.colorbar(c)
    ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
    ax.set_ylabel(r'$\phi$')
    ax.set_xlabel(r'$\theta$')
    plt.savefig(os.path.join(figure_path, 'normal_field_error_no_wp.png'))


    
    # Run and plot Poincare section
    vmec_surf = SurfaceRZFourier.from_wout(v.output_file) # vmec surface
    vmec_surf_1 = SurfaceRZFourier.from_wout(v.output_file) # Expanded vmec surface
    vmec_surf_1.extend_via_normal(0.1)
    nfp = vmec_surf.nfp
    
    Rmaj = vmec_surf.major_radius()
    r0 = vmec_surf.minor_radius()
    sc_fieldline = SurfaceClassifier(vmec_surf_1, h=0.01, p=3)
    nfieldlines = 50
    tmax_fl = 2500
    degree = 4
    
    def trace_fieldlines(bfield,label):
        # Set up initial conditions - 
        R0 = np.linspace(Rmaj-2*r0, Rmaj+2*r0, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-8,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(figure_path, 'poincare_no_wp'), dpi=150,surf=vmec_surf,mark_lost=True)
        return fieldlines_phi_hits
    
    hits = trace_fieldlines(bs, 'vmec')









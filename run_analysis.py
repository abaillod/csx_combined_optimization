import booz_xform as bx
from simsopt.mhd.vmec import Vmec
import argparse
import os
from simsopt._core.optimizable import load, save
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data


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


v = Vmec('input.final')
v.run()


# Run boozer
b = bx.Booz_xform()
b.read_wout(v.output_file)
b.mboz = 48
b.nboz = 48
b.run()
b.write_boozmn("boozmn.nc")

bmnc = b.bmnc_b
xm = b.xm_b
xn = b.xn_b
f = np.sqrt(np.sum(bmnc[xn!=0,:]**2,axis=0)/np.sum(bmnc**2,axis=0))
print('Mean QA metric: ',np.mean(f))
print('Mean iota: ',np.mean(b.iota))
s = b.s_b

# Plot quaissymmetry
plt.figure()
plt.plot(s,f)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('f [quasisymmetry metric]')
plt.savefig(os.path.join(figure_path, 'fquasisymmetry'))
if args.show:
    plt.show()

# Plot rotational transform
plt.figure()
plt.plot(s,b.iota)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('iota')
plt.savefig(os.path.join(figure_path, 'iota'))
if args.show:
    plt.show()

# Plot |B| contours
plt.figure()
bx.surfplot(b, js=-1)
plt.savefig(os.path.join(figure_path, 'modB.png'))
if args.show:
    plt.show()

# plot magnetic well
plt.figure()
plt.plot(v.s_half_grid,v.wout.vp[1::])
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('Vp [radial derivative of volume]')
plt.savefig(os.path.join(figure_path, 'magnetic_well'))
if args.show:
    plt.show()

# Plot B.n
bs = load(os.path.join(this_path, 'coils/bs_output.json'))
surf = v.boundary
theta = surf.quadpoints_theta
phi = surf.quadpoints_phi
ntheta = theta.size
nphi = phi.size
bs.set_points(surf.gamma().reshape((-1,3)))
Bdotn = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)

fig, ax = plt.subplots(figsize=(12,5))
c = ax.contourf(theta,phi,Bdotn)
plt.colorbar(c)
ax.set_title(r'Initial $\mathbf{B}\cdot\hat{n}$ on CSSC surface')
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$\theta$')
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
    plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(figure_path, 'poincare'), dpi=150,surf=vmec_surf,mark_lost=False)
    return fieldlines_phi_hits

hits = trace_fieldlines(bs, 'vmec')


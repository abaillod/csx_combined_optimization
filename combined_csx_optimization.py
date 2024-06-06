"""
Combined approach
-----------------
CSX coil optimization based on the combined approach proposed by R. Jorge et. al. (2023). 
https://arxiv.org/abs/2302.10622

The main idea is to combined a fixed-boundary VMEC calculation with coil optimization. Degrees of 
freedom are then the coils geometry and current, and the VMEC plasma boundary harmonics. Target 
includes self-consistency between coils and VMEC boundary (i.e. the quadratic flux across the 
plasma boundary should be zero), plasma target functions (iota, QA, aspect ratio, ...) and coil 
engineering target (max curvature, length, torsion, HTS constraints, ...)

We consider two circular poloidal field coils, harvested from the former CNT device, and optimize 
two interlinked coils and some planar windowpane coils.

Usage:: run with
```
python combined_csx_optimization.py --input path/to/input [--options]
```
where options are:
    --pickle, if the input file is a pickle file
"""
# Import and metadata
# -------------------
import simsopt
import sys
import importlib
import os
import datetime
import numpy as np
import pickle
from mpi4py import MPI
from pathlib import Path
from pystellplot.Paraview import coils_to_vtk, surf_to_vtk

from simsopt.field.biotsavart import BiotSavart
from simsopt._core.optimizable import load
from scipy.optimize import minimize
#from scipy.interpolate import interp1d
from simsopt._core import Optimizable
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, WellWeighted
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries, apply_symmetries_to_curves
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem, Weight
from simsopt.geo import CurveLength, CurveCurveDistance, MeanSquaredCurvature,  LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves,  CurveSurfaceDistance
from simsopt.geo.orientedcurve import OrientedCurveXYZFourier
from simsopt.field.coilobjective import CurrentPenalty
from simsopt.field.coil import apply_symmetries_to_currents, ScaledCurrent
from simsopt.field.coil import Coil
from set_default_values import set_default
from simsopt._core.util import ObjectiveFailure
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import comm_world

from simsopt.geo.framedcurve import FramedCurveFrenet, FramedCurveCentroid
from simsopt.geo import FramedCurveTwist, CoilStrain, LPTorsionalStrainPenalty, LPBinormalCurvatureStrainPenalty, FrameRotation

import git
import argparse

from jax import grad
import jax.numpy as jnp
from simsopt.geo.jit import jit
from simsopt._core.derivative import derivative_dec

from vacuum_vessel import CSX_VacuumVessel, VesselConstraint

# Setup MPI
mpi = MpiPartition()

# Read command line arguments
parser = argparse.ArgumentParser()

# If ran with "--pickle", expect the input to be a pickle.
parser.add_argument("--pickle", dest="pickle", default=False, action="store_true")

# Provide input as a relative or absolute path
parser.add_argument("--input", dest="input", default=None)

# Prepare args
args = parser.parse_args()

# ====================================================================================================
# INITIALIZATION
# --------------

# Paths
parent_path = str(Path(__file__).parent.resolve()) 
os.chdir(parent_path)
print('parent_path: ',parent_path)

# Read input - if first line argument is 0, read a python file; if first line argument is 1, 
# read a pickle. Otherwise, raise an error.
if args.pickle:
    with open(args.input, 'rb') as f:
        inputs = pickle.load(f)
else:
    fname = args.input.replace('/','.')
    if fname[-3:]=='.py':
        fname = fname[:-3]
    std = importlib.import_module(fname, package=None)
    inputs = std.inputs

# Set defaut values
set_default_str = set_default(inputs)

# Create output directories. Use directory name provided in input file; if not provided, create
# a generic name using the date and time.
date = datetime.datetime
if 'directory' in inputs.keys():
    dir_name = inputs['directory']
else:
    dir_name = 'runs/' + date.now().isoformat(timespec='seconds') + '/'

# If directory already exist, crash. We don't want to loose optimization results we obtained in a former run!
this_path = os.path.join(parent_path, dir_name)
if comm_world.rank == 0:
    os.makedirs( this_path, exist_ok=False )
MPI.COMM_WORLD.Barrier()
os.chdir(this_path)

# Create a few more paths, and move in result directory. This is useful so that all output files produced by simsopt
# are all stored in the same location
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0: 
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

    # Create log file, define function to append to log file
    #repo = git.Repo('/burg/home/ab5667/Github/simsopt', search_parent_directories=True)
    repo = git.Repo('~/Github/simsopt', search_parent_directories=True)
    sha0 = repo.head.object.hexsha
    
    repo = git.Repo(search_parent_directories=True)
    sha1 = repo.head.object.hexsha
    with open(os.path.join(this_path,'log.txt'), 'w') as f:
        f.write("CSX COMBINED OPTIMIZATION\n")
        f.write(f"Using simsopt version {sha0}\n")
        f.write(f"Using csx optimization git version {sha1}\n")
        f.write(f"Date = {date.date(date.now()).isoformat()} at {date.now().strftime('%Hh%M')}\n")
        f.write(set_default_str)

def log_print(mystr):
    """Print into log file

    Args:
        - mystr: String to be printed
        - first: Set to True to create log file. 
    """
    if comm_world.rank == 0: 
        with open(os.path.join(this_path,'log.txt'), 'a') as f:
            f.write(mystr)

# Save input
# We save both the input as a pickle (to be called again to repeat the optimization), and as
# a text file, for a quick vizualization.
if comm_world.rank == 0: 
    with open(os.path.join(this_path, 'input.pckl'), 'wb') as f:
        pickle.dump(inputs, f)

def print_dict_recursive(file, d, order=0, k=None):
    """Recursive print routine to print a dictionary"""
    if type(d) is dict:
        for k, i in d.items():
            if type(i) is dict:
                for l in range(order):
                    file.write('\n')
                for l in range(order+1):
                    file.write('#')
                if order>0:
                    file.write(' ')
                file.write(f'{k}\n')
            print_dict_recursive(file, i, order=order+1, k=k)
            file.write(f' \n')
    elif type(d) is Weight:
        file.write(f'{k} = {d.value}')
    else:
        for l in range(order-1):
            file.write('')
        file.write(f'{k} = {d}')

if comm_world.rank == 0: 
    with open(os.path.join(this_path, 'input.txt'), 'w') as f:
        print_dict_recursive(f, inputs)


# =================================================================================================
# CREATE INITIAL COILS AND SURFACE
# --------------------------------
# In this section we prepare all the objects required by the optimization, namely the coils, the plasma
# boundary, and the Vmec instance.

# Load Vmec object, extract the boundary
vmec = Vmec(
    os.path.join( parent_path, inputs['vmec']['filename'] ),
    mpi=mpi, 
    verbose=inputs['vmec']['verbose'], 
    nphi=inputs['vmec']['nphi'], 
    ntheta=inputs['vmec']['ntheta']
)
vmec.indata.mpol = inputs['vmec']['internal_mpol'] 
vmec.indata.ntor = inputs['vmec']['internal_ntor'] 
surf = vmec.boundary


max_boundary_mpol = inputs['vmec']['max_boundary_mpol']
if max_boundary_mpol is None:
    max_boundary_mpol = inputs['vmec']['internal_mpol'] 
    
max_boundary_ntor = inputs['vmec']['max_boundary_ntor']
if max_boundary_ntor is None:
    max_boundary_ntor = inputs['vmec']['internal_ntor'] 
    
for mm in range(max_boundary_mpol+1, surf.mpol+1):
    for nn in range(-surf.ntor, surf.ntor+1):
        surf.set(f'rc({mm},{nn})', 0)
        surf.set(f'zs({mm},{nn})', 0)
for nn in range(max_boundary_ntor+1, surf.ntor+1):
    for mm in range(0, surf.mpol+1):
        for p_or_m in [-1,1]:
            if mm==0 and p_or_m==-1:
                continue
            surf.set(f'rc({mm},{p_or_m*nn})', 0)

            if mm==0 and nn==0:
                continue
            surf.set(f'zs({mm},{p_or_m*nn})', 0)
            
        
# Save initial vmec
vmec.write_input(os.path.join(this_path, f'input.initial'))

# Load IL and PF initial coils. Extract the base curves and currents.
print(f"Loading the coils from file {os.path.join(parent_path, inputs['cnt_coils']['geometry']['filename'])}")
bs = load( os.path.join(parent_path, inputs['cnt_coils']['geometry']['filename']) )
cnt_initial_coils = bs.coils
#il_base_coil = cnt_initial_coils.coils[0]
#il_coils = cnt_initial_coils.coils[0:2]
#pf_base_coil = cnt_initial_coils.coils[2]
#pf_coils = cnt_initial_coils.coils[2:4]

# Renormalize currrents
base_il_current = Current( 1 ) # Dof is now order 1
base_il_current.name = 'IL_current'
base_pf_current = Current( 1 ) # Dof is now order 1
base_pf_current.name = 'PF_current'

il_current = cnt_initial_coils[0].current.get_value()
if cnt_initial_coils[1].current.get_value() == il_current:
    il_sgn = +1
else:
    il_sgn = -1

pf_current = cnt_initial_coils[2].current.get_value()
if cnt_initial_coils[3].current.get_value() == pf_current:
    pf_sgn = +1
else:
    pf_sgn = -1

c0 = Coil( cnt_initial_coils[0].curve, ScaledCurrent( base_il_current, il_current ) )
c1 = Coil( cnt_initial_coils[1].curve, ScaledCurrent( base_il_current, il_sgn*il_current ) )
c2 = Coil( cnt_initial_coils[2].curve, ScaledCurrent( base_pf_current, pf_current ) )
c3 = Coil( cnt_initial_coils[3].curve, ScaledCurrent( base_pf_current, pf_sgn*pf_current ) )
il_base_coil = c0
il_coils = [c0, c1]
pf_base_coil = c2
pf_coils = [c2, c3]

# Remove this to free some memory...
del(cnt_initial_coils)
del(bs)

# Extract core curves. Rename each coil for easier reading of the dofs name.
il_curve = il_coils[0].curve

il_base_current = il_coils[0].current
il_base_current.name = 'IL_base_current'

pf_curve = pf_coils[0].curve
pf_base_curve = pf_curve
while hasattr(pf_base_curve, 'curve'):
    pf_base_curve = pf_base_curve.curve
pf_base_curve.name = 'PF_base_curve'

pf_base_current = pf_coils[0].current
pf_base_current.name = 'PF_base_current'


# Create curve frame
rotation = FrameRotation(il_curve.quadpoints, inputs['winding']['rot_order'])

fc_centroid = FramedCurveCentroid(il_curve)
fc_frenet = FramedCurveFrenet(il_curve)
fc = FramedCurveCentroid(il_curve,rotation)

twist = FramedCurveTwist(fc)
cs = CoilStrain(fc, width=inputs['winding']['width'])

# Load or generate windowpane coils
if inputs['wp_coils']['geometry']['filename'] is None:
    if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
        wp_base_curves = []
        wp_base_currents = []
        for Z0 in inputs['wp_coils']['geometry']['Z0']:
            wp_base_curves += create_equally_spaced_windowpane_curves( 
                inputs['wp_coils']['geometry']['ncoil_per_row'], 
                surf.nfp, surf.stellsym, 
                inputs['wp_coils']['geometry']['R0'], 
                inputs['wp_coils']['geometry']['R1'], 
                Z0, order=10
            )
            
        wp_base_currents += [ScaledCurrent( Current(0), 1e5 ) for c in wp_base_curves]
        wp_base_coils = [Coil(curve, current) for curve, current in zip(wp_base_curves, wp_base_currents)]
        wp_coils = [Coil(curve,current) for curve, current in zip(
            apply_symmetries_to_curves(wp_base_curves, surf.nfp, surf.stellsym),
            apply_symmetries_to_currents(wp_base_currents, surf.nfp, surf.stellsym)
        )]
    
    else:
        wp_coils = []
        wp_base_coils = []
        wp_base_curves = []
        wp_base_currents = []

else:
    bs = load( os.path.join(parent_path, inputs['wp_coils']['geometry']['filename']) )
    wp_coils = bs.coils
    nwp_base = inputs['wp_coils']['geometry']['n_base_coils']
    if nwp_base is None:
        raise ValueError('Need to provide number of base WP coils')
    wp_base_coils = bs.coils[:nwp_base]
    wp_base_curves = [c.curve for c in wp_base_coils]
    wp_base_currents = [c.current for c in wp_base_coils]

for ii, c in enumerate(wp_base_curves):
    c.name = f'WP_base_curve_{ii}'
for ii, c in enumerate(wp_base_currents):
    c.name = f'WP_base_current_{ii}'

# Define some useful arrays
full_coils = il_coils + pf_coils + wp_coils
full_curves = [c.curve for c in full_coils]

base_coils = [il_base_coil, pf_base_coil] + wp_base_coils
base_curves = [c.curve for c in base_coils]

# Define the BiotSavart field and set evaluation points on the VMEC boundary
bs = BiotSavart(full_coils)
bs_wp = BiotSavart(wp_coils) # just for output
bs.set_points( surf.gamma().reshape((-1,3)) )

# Save initial coils and surface
if comm_world.rank==0:
    coils_to_vtk( full_coils, os.path.join(coils_results_path, "initial_coils") )
    surf_to_vtk( os.path.join(coils_results_path, "initial_surface"), bs, surf )
    bs.save( os.path.join(coils_results_path, "bs_initial.json") )
    bs_wp.save( os.path.join(coils_results_path, "bs_wp_initial.json") )
    fc.save( os.path.join(coils_results_path, "hts_frame_initial.json") )


# =================================================================================================
# DEFINE NEW PENALTIES

@jit
def Lp_R_pure(gamma, gammadash, p, Rmax):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    R = jnp.sqrt(gamma[:,1]**2 + gamma[:,2]**2)
    return (1./p)*jnp.mean(jnp.maximum(R-Rmax, 0)**p * arc_length)


class LpCurveR(Optimizable):
    r"""
    This class computes a penalty term based on the maximum R position of a curve.
    Used to constrain the coil to remain within a cylindrical vessel
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda gamma, gammadash: Lp_R_pure(gamma, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda gamma, gammadash: grad(self.J_jax, argnums=0)(gamma, gammadash))
        self.thisgrad1 = jit(lambda gamma, gammadash: grad(self.J_jax, argnums=1)(gamma, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.gamma(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.gamma(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.gamma(), self.curve.gammadash())
        
        return self.curve.dgamma_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}

@jit
def Lp_Z_pure(gamma, gammadash, p, Zmax):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    Z = gamma[:,0]
    return (1./p)*jnp.mean(jnp.maximum(Z-Zmax, 0)**p * arc_length)


class LpCurveZ(Optimizable):
    r"""
    This class computes a penalty term based on the maximum |Z| position of a curve.
    Used to constrain the coil to remain within a cylindrical vessel
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda gamma, gammadash: Lp_Z_pure(gamma, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda gamma, gammadash: grad(self.J_jax, argnums=0)(gamma, gammadash))
        self.thisgrad1 = jit(lambda gamma, gammadash: grad(self.J_jax, argnums=1)(gamma, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.gamma(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.gamma(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.gamma(), self.curve.gammadash())
        
        return self.curve.dgamma_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}



# =================================================================================================
# RUN STAGE TWO OPTIMIZATION
# --------------------------------
# We begin with a stage two optimization to get the coils as close as possible to the VMEC 
# boundary. Here, we only include coils penalty function and attempt at minimizing the quadratic
# flux across VMEC boundary.
square_flux = SquaredFlux(surf, bs, definition="local")
square_flux_threshold = inputs['cnt_coils']['target']['square_flux_threshold']
square_flux_penalty_type = inputs['cnt_coils']['target']['square_flux_constraint_type']
if square_flux_penalty_type=='objective':
    Jcoils = square_flux
elif square_flux_penalty_type=='max':
    Jcoils = QuadraticPenalty( square_flux, square_flux_threshold, 'max' )
else:
    raise ValueError('Invalid square flux penalty type')

def add_target(Jcoils, J, w):
    # This is a small wrapper to avoid adding some 0*J() to the target function.
    # I don't know if it might cause numerical issues.
    if w.value>0:
        Jcoils += w * J
    return Jcoils

# IL-coils penalties
il_length = CurveLength( il_curve )
il_length_target = inputs['cnt_coils']['target']['IL_length']
il_length_penalty_type = inputs['cnt_coils']['target']['IL_length_constraint_type']
il_length_weight = inputs['cnt_coils']['target']['IL_length_weight'] 
Jcoils = add_target(Jcoils, QuadraticPenalty( il_length, il_length_target, il_length_penalty_type ), il_length_weight)

il_curvature_threshold = inputs['cnt_coils']['target']['IL_maxc_threshold']
il_curvature_weight = inputs['cnt_coils']['target']['IL_maxc_weight']
il_curvature = LpCurveCurvature(il_curve, 2, il_curvature_threshold)
Jcoils = add_target( Jcoils, il_curvature, il_curvature_weight )

il_msc = MeanSquaredCurvature( il_curve )
il_msc_threshold = inputs['cnt_coils']['target']['IL_msc_threshold']
il_msc_weight = inputs['cnt_coils']['target']['IL_msc_weight']
Jcoils = add_target( Jcoils, QuadraticPenalty(il_msc, il_msc_threshold, f='max'), il_msc_weight )

il_curveR_threshold = inputs['cnt_coils']['target']['IL_maxR_threshold'] 
il_curveR_weight = inputs['cnt_coils']['target']['IL_maxR_weight']
Jcoils = add_target( Jcoils, LpCurveR( il_curve, 2, il_curveR_threshold ), il_curveR_weight )

il_curveZ_threshold = inputs['cnt_coils']['target']['IL_maxZ_threshold'] 
il_curveZ_weight = inputs['cnt_coils']['target']['IL_maxZ_weight']
Jcoils = add_target( Jcoils, LpCurveZ( il_curve, 2, il_curveZ_threshold ), il_curveZ_weight )

il_arclength_weight = inputs['cnt_coils']['target']['arclength_weight'] 
Jcoils = add_target( Jcoils, ArclengthVariation( il_curve ), il_arclength_weight )

il_tor_weight = inputs['winding']['il_tor_weight'] 
Jcoils = add_target( Jcoils, LPTorsionalStrainPenalty(fc, p=2, threshold=inputs['winding']['tor_threshold'], width=inputs['winding']['width']), il_tor_weight )

il_bincurv_weight = inputs['winding']['il_bincurv_weight'] 
Jcoils = add_target( Jcoils, LPBinormalCurvatureStrainPenalty(fc, p=2, threshold=inputs['winding']['cur_threshold'], width=inputs['winding']['width']), il_bincurv_weight )

il_twist_weight = inputs['winding']['il_twist_weight']
Jcoils = add_target( Jcoils, QuadraticPenalty(twist,inputs['winding']['il_twist_max'],'max'), il_twist_weight )


# WP penalties
if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
    wp_lengths = [CurveLength( c ) for c in wp_base_curves]
    wp_length_threshold = inputs['wp_coils']['target']['length']
    wp_length_penalty_type = inputs['wp_coils']['target']['length_constraint_type']
    wp_length_weight = inputs['wp_coils']['target']['length_weight']
    Jcoils = add_target( 
        Jcoils, 
        sum([QuadraticPenalty( wpl, wp_length_threshold, wp_length_penalty_type) for wpl in wp_lengths]),
        wp_length_weight
    )

    wp_curvature_threshold = inputs['wp_coils']['target']['maxc_threshold']
    wp_curvature_weight = inputs['wp_coils']['target']['maxc_weight']
    wp_curvatures = [LpCurveCurvature(c, 2, wp_curvature_threshold) for c in wp_base_curves]
    Jcoils = add_target( Jcoils, sum(wp_curvatures), wp_curvature_weight )

    wp_msc = [MeanSquaredCurvature(c) for c in wp_base_curves]
    wp_msc_threshold = inputs['wp_coils']['target']['msc_threshold']
    wp_msc_weight = inputs['wp_coils']['target']['msc_weight']
    Jcoils = add_target( Jcoils, sum([QuadraticPenalty(msc, wp_msc_threshold, f='max') for msc in wp_msc]), wp_msc_weight )
    
    wp_maxZ_threshold = inputs['wp_coils']['target']['WP_maxZ_threshold']
    wp_maxZ_weight = inputs['wp_coils']['target']['WP_maxZ_weight']
    J_maxZ_wp = sum([LpCurveZ( c, 2, wp_maxZ_threshold ) for c in wp_base_curves])
    Jcoils = add_target( Jcoils, J_maxZ_wp, wp_maxZ_weight )
 

il_vessel_threshold = inputs['cnt_coils']['target']['IL_vessel_threshold'] 
il_vessel_weight = inputs['cnt_coils']['target']['IL_vessel_weight']
if il_vessel_threshold<0 and il_vessel_weight.value!=0:
    raise ValueError('il_vessel_threshold should be greater than 0!')
vessel = CSX_VacuumVessel()
vpenalty = VesselConstraint( [il_curve] + wp_base_curves, vessel, il_vessel_threshold )
Jcoils = add_target( Jcoils, vpenalty, il_vessel_weight )

Jccdist = CurveCurveDistance(full_curves, inputs['CC_THRESHOLD'], num_basecurves=len(full_curves))
Jcoils = add_target( Jcoils, Jccdist, inputs['CC_WEIGHT'] ) 

Jcsdist = CurveSurfaceDistance([il_curve], surf, inputs['CS_THRESHOLD'])
Jcoils = add_target( Jcoils, Jcsdist, inputs['CS_WEIGHT'] ) 

def fun_coils(dofs, info, verbose=True):
    """Objective function for the stage II optimization

    Args:
        - dofs: Coils degrees of freedom. Should have the same size as Jcoils.x
        - info: Dictionary with key "Nfeval' - used as a number of function evaluation counter
    Outputs:
        - J: Objective function value 
        - grad: Derivative of J w.r.t the dofs
    """
    info['Nfeval'] += 1 # Increase counter
    Jcoils.x = dofs     # Set new dofs
    J = Jcoils.J()      # Evaluate objective function
    grad = Jcoils.dJ()  # Evaluate gradient

    # Prepare string output for log file
    if mpi.proc0_world:
        sqf = square_flux.J()
        nphi_VMEC = vmec.boundary.quadpoints_phi.size
        ntheta_VMEC = vmec.boundary.quadpoints_theta.size
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        VP = vpenalty.J()
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, square_flux={sqf:.1e}, ⟨B·n⟩={BdotN:.1e}" 
        outstr += f", ║∇J coils║={np.linalg.norm(Jcoils.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f"IL length={il_length.J():.2f},  IL ∫ϰ²/L={il_msc.J():.2f},  IL ∫max(ϰ-ϰ0,0)^2={il_curvature.J():.2f}\n"
        outstr += f"Vessel penalty is {VP:.2E}\n"
        outstr += f"HTS:: torsional strain={np.max(cs.torsional_strain()):.2E}, curvature strain={np.max(cs.binormal_curvature_strain()):.2E}, frame twist={twist.J():.2E}\n"
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
            for i, (l, msc, jcs) in enumerate(zip(wp_lengths, wp_msc, wp_curvatures)):
                outstr += f"WP_{i:d} length={l.J():.2f},  WP_{i:d} ∫ϰ²/L={msc.J():.2f},  WP_{i:d} ∫max(ϰ-ϰ0,0)^2={jcs.J():.2f}\n" 
            outstr += f"\n"

        if verbose:
            log_print(outstr)

    return J, grad


# Define QS metric. Here we target QS (M=1, N=0)
qs = QuasisymmetryRatioResidual(
        vmec, inputs['vmec']['target']['qa_surface'], helicity_m=1, helicity_n=0, 
        ntheta=inputs['vmec']['target']['qa_ntheta'], nphi=inputs['vmec']['target']['qa_nphi']
)

class remake_iota(Optimizable):
    """ Penalty function for the mean value of iota. 

    This is useful to use the QuadraticPenalty function of simsopt.

    Args:
        - vmec: simsopt.mhd.Vmec instance
    """
    def __init__(self, vmec):
        self.vmec = vmec
        super().__init__(depends_on=[vmec])
    def J(self):
        try:
            return self.vmec.mean_iota()
        except ObjectiveFailure: 
            log_print(f"Error evaluating iota! ")
            return np.nan

class remake_aspect(Optimizable):
    """ Penalty function for the aspect ratio. 

    This is useful to use the QuadraticPenalty function of simsopt.

    Args:
        - vmec: simsopt.mhd.Vmec instance
    """
    def __init__(self, vmec):
        self.vmec = vmec
        super().__init__(depends_on=[vmec])
    def J(self):
        "returns value of quantity"
        try: 
            return self.vmec.aspect()
        except ObjectiveFailure: 
            log_print(f"Error evaluating aspect ratio! ")
            return np.nan

class quasisymmetry(Optimizable):
    def __init__(self, qs):
        self.qs = qs  
        super().__init__(depends_on=[qs])
    def J(self):
        "returns value of quantity" 
        try:
            return self.qs.total()
        except ObjectiveFailure: 
            with open(os.path.join(this_path, 'log.txt'), 'a') as f:
                f.write(f"Error evaluating QS! ")
            return np.nan

class volume(Optimizable):
    def __init__(self, surf):
        self.surf = surf
        super().__init__(depends_on=[surf])
    def J(self):
        try:
            return self.surf.volume()
        except ObjectiveFailure:
            with open(os.path.join(this_path, 'log.txt'), 'a') as f:
                f.write(f"Error evaluating Volume! ")
            return np.nan
    
# class IntervalWell(Optimizable):
#     def __init__(self, vmec, smin, smax):
#         self.vmec = vmec
#         self.smin = smin
#         self.smax = smax
#         super().__init__(depends_on = [vmec])
#     def J(self):
#         self.vmec.run()
#         smax = self.smax
#         smin = self.smin
#         dVds = 4 * np.pi * np.pi * np.abs(self.vmec.wout.gmnc.T[1:, 0])
#         dVds_interp = interp1d(self.vmec.s_half_grid, dVds, fill_value='extrapolate')
#         d2_V_d_s2_avg = (dVds_interp(smax) - dVds_interp(smin)) / (smax - smin)
#         interval_well = -d2_V_d_s2_avg / (0.5 * (dVds_interp(smax) + dVds_interp(smin)))
#         return interval_well        


J_iota = inputs['vmec']['target']['iota_weight'] * QuadraticPenalty(remake_iota(vmec), inputs['vmec']['target']['iota'], inputs['vmec']['target']['iota_constraint_type'])
J_aspect = inputs['vmec']['target']['aspect_ratio_weight'] * QuadraticPenalty(remake_aspect(vmec), inputs['vmec']['target']['aspect_ratio'], inputs['vmec']['target']['aspect_ratio_constraint_type'])
J_qs = QuadraticPenalty(quasisymmetry(qs), 0, 'identity') 
J_volume =  inputs['vmec']['target']['volume_weight'] * QuadraticPenalty( volume( surf ),  inputs['vmec']['target']['volume'],  inputs['vmec']['target']['volume_constraint_type'] )

#if inputs['vmec']['target']['magnetic_well_type']=='weighted':
#    weight1 = lambda s: np.exp(-s**2/0.01**2)
#    weight2 = lambda s: np.exp(-(1-s)**2/0.01**2)
#    J_well = inputs['vmec']['target']['magnetic_well_weight'] * WellWeighted( vmec, weight1, weight2 )
#elif inputs['vmec']['target']['magnetic_well_type']=='standard':
#    J_well =IntervalWell(vmec, 0.2, 0.4) + IntervalWell(vmec, 0.8, 0.99)

Jplasma = J_qs
# Only add targets with non-zero weight.
if inputs['vmec']['target']['iota_weight'].value>0:
    Jplasma += J_iota
if inputs['vmec']['target']['aspect_ratio_weight'].value>0:
    Jplasma += J_aspect
if inputs['vmec']['target']['volume_weight'].value>0:
    Jplasma += J_volume
#if inputs['vmec']['target']['magnetic_well_weight'].value>0:
#    Jplasma += J_well





# We now include both the coil penalty and the plasma target functions
outputs = dict()
outputs['J'] = []                   # Full target function
outputs['dJ'] = []                  # Jacobian
outputs['Jplasma'] = []             # Plasma target function
outputs['dJplasma'] = []            # Jacobian of plasma target
outputs['Jcoils'] = []              # Coil target function
outputs['dJcoils'] = []             # Jacobian of coil target
outputs['iota_axis'] = []           # Iota on axis, as evaluated by VMEC
outputs['iota_edge'] = []           # Iota at the edge, as evaluated by VMEC
outputs['mean_iota'] = []           # Mean iota, as evaluated by VMEC
outputs['aspect'] = []              # Aspect ratio
outputs['QSresiduals'] = []         # QS residuals
outputs['QSprofile'] = []           # QS profile
outputs['QuadFlux'] = []            # Quadratic flux through plasma boundary
outputs['BdotN'] = []               # Value of B.n/|n| evaluated on the plasma boundary grid
outputs['min_CS'] = []              # Min plasma-coil distance
outputs['min_CC'] = []              # Min coil-coil distance
outputs['IL_length'] = []           # Length of IL coil
outputs['WP_length'] = []           # Length of WP coils
outputs['IL_msc'] = []              # Mean square curvature of IL coil
outputs['WP_msc'] = []              # Mean square curvature of WP coils
outputs['IL_max_curvature'] = []    # IL max curvature penalty. This is 0 if below threshold 
outputs['WP_max_curvature'] = []    # WP max curvature penalty. This is 0 if below threshold
outputs['vessel_penalty'] = []    
outputs['vmec'] = dict()
outputs['vmec']['fsqr'] = []        # Force balance error in VMEC, radial direction ?
outputs['vmec']['fsqz'] = []        # Force balance error in VMEC, vertical direction ?
outputs['vmec']['fsql'] = []        # Force balance error in VMEC, ??? direction ?
outputs['torsional_strain'] = []
outputs['curvature_strain'] = []
outputs['frame_twist'] = []


def set_dofs(x0):
    """ Set the degrees of freedom of the coils and the plasma boundary

    Args:
        - x0: np.array of size Jcoils.x.size + vmec.x.size
    """
    # Check if there are any difference between Jcoils.x and the new dofs. 
    # If there are, replace with new values. This calls internally the 
    # routines "recompute_bell", informing simsopt that all objectives
    # have to be reevaluated.
    if np.sum(Jcoils.x!=x0[:-ndof_vmec])>0:
        Jcoils.x = x0[:-ndof_vmec]

    # Same for the plasma dofs...
    if np.sum(Jplasma.x!=x0[-ndof_vmec:])>0:
        Jplasma.x = x0[-ndof_vmec:]

    # Update the Biotsavart field evaluation points
    bs.set_points(surf.gamma().reshape((-1, 3)))


# Define target function
JACOBIAN_THRESHOLD = inputs['numerics']['JACOBIAN_THRESHOLD']
def fun(dofs, prob_jacobian=None, info={'Nfeval':0}, verbose=True):
    info['Nfeval'] += 1
    coils_objective_weight = inputs['coils_objective_weight']
    
    # Set the dofs
    set_dofs(dofs)
    
    # Evaluate target function
    os.chdir(vmec_results_path)
    J_stage_1 = Jplasma.J()
    J_stage_2 = coils_objective_weight.value * Jcoils.J()
    J = J_stage_1 + J_stage_2

    outputs['J'].append(float(J))
    outputs['Jplasma'].append(float(J_stage_1))
    outputs['Jcoils'].append(float(J_stage_2))
    outputs['vmec']['fsqr'].append(vmec.wout.fsqr)
    outputs['vmec']['fsqz'].append(vmec.wout.fsqz)
    outputs['vmec']['fsql'].append(vmec.wout.fsql)
        
    if J > inputs['numerics']['JACOBIAN_THRESHOLD'] or np.isnan(J):
        log_print(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}\n")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * ndof_vmec
        grad_with_respect_to_coils = [0] * len(Jcoils.x)
        outstr = f"STEP {info['Nfeval']:03.0f}. J_stage_1 = {J_stage_1}, J_stage_2 = {J_stage_2}\n"
        outstr += "VMEC FAILED\n"

        # Append each output array with a np.nan
        outputs['dJplasma'].append(np.nan)
        outputs['dJcoils'].append(np.nan)
        outputs['iota_axis'].append(np.nan)
        outputs['iota_edge'].append(np.nan)
        outputs['mean_iota'].append(np.nan)
        outputs['aspect'].append(np.nan)
        outputs['QSresiduals'].append(np.nan)
        outputs['QSprofile'].append(np.nan)
        outputs['QuadFlux'].append(np.nan)
        outputs['BdotN'].append(np.nan)
        outputs['min_CS'].append(np.nan)
        outputs['min_CC'].append(np.nan)
        outputs['IL_length'].append(np.nan)
        outputs['vessel_penalty'].append(np.nan)
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0: 
            outputs['WP_length'].append([np.nan for l in wp_lengths])
            outputs['WP_msc'].append([np.nan for msc in wp_msc])
            outputs['WP_max_curvature'].append([np.nan for c in wp_curvatures])
        outputs['IL_msc'].append(np.nan)
        outputs['IL_max_curvature'].append(np.nan)
        outputs['torsional_strain'].append(np.nan)
        outputs['curvature_strain'].append(np.nan)
        outputs['frame_twist'].append(np.nan)

    else:
        # Evaluate important metrics    
        n = surf.normal() # Plasma boundary normal
        absn = np.linalg.norm(n, axis=2) 
        
        nphi_VMEC = surf.quadpoints_phi.size
        ntheta_VMEC = surf.quadpoints_theta.size
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        
        Bcoil = bs.B().reshape(n.shape)
        unitn = n / absn[:, :, None]
        B_n = np.sum(Bcoil*unitn, axis=2)     # This is B.n/|n|
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2) # This is |B|
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2) # This is B.n

        # Save in output arrays
        outputs['iota_axis'].append(float(vmec.iota_axis()))
        outputs['iota_edge'].append(float(vmec.iota_edge()))
        outputs['mean_iota'].append(float(vmec.mean_iota()))
        outputs['aspect'].append(float(vmec.aspect()))
        outputs['QSresiduals'].append(np.array(qs.residuals()))
        outputs['QSprofile'].append(np.array(qs.profile()))
        outputs['QuadFlux'].append(float(square_flux.J()))
        outputs['BdotN'].append(np.array(B_n))
        outputs['min_CS'].append(float(Jcsdist.shortest_distance()))
        outputs['min_CC'].append(float(Jccdist.shortest_distance()))
        outputs['IL_length'].append(float(il_length.J()))
        outputs['vessel_penalty'].append(float(vpenalty.J()))
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0: 
            outputs['WP_length'].append([float(l.J()) for l in wp_lengths])
            outputs['WP_msc'].append([float(msc.J()) for msc in wp_msc])
            outputs['WP_max_curvature'].append([float(c.J()) for c in wp_curvatures])
        outputs['IL_msc'].append(float(il_msc.J()))
        outputs['IL_max_curvature'].append(float(il_curvature.J()))
        outputs['torsional_strain'].append(cs.torsional_strain())
        outputs['curvature_strain'].append(cs.binormal_curvature_strain())
        outputs['frame_twist'].append(twist.J())

   
        # Log output
        outstr = f"STEP {info['Nfeval']:03.0f}. J_stage_1 = {J_stage_1}, J_stage_2 = {J_stage_2}\n"
        outstr += f"aspect={outputs['aspect'][-1]:.5E}, iota_axis={outputs['iota_axis'][-1]:.5E}, iota_edge={outputs['iota_edge'][-1]:.5E}, iota={outputs['mean_iota'][-1]:.5E}\n"
        outstr += f"QS profile="
        for o in outputs['QSprofile'][-1]:
            outstr += f"{o:.5E}, "
        outstr += f"\n square_flux={outputs['QuadFlux'][-1]:.5E}, ⟨B·n⟩={np.mean(np.abs(B_n)):.5E}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", C-C-Sep={outputs['min_CC'][-1]:.5E}"
        outstr += f", C-S-Sep={outputs['min_CS'][-1]:.5E}"
        outstr += f", IL length={outputs['IL_length'][-1]:.5E},  IL ∫ϰ²/L={outputs['IL_msc'][-1]:.5E},  IL ∫max(ϰ-ϰ0,0)^2={outputs['IL_max_curvature'][-1]:.5E}\n"
        outstr += f"Vessel penalty is {outputs['vessel_penalty'][-1]:.2E}\n"
        outstr += f"HTS:: torsional strain={np.max(outputs['torsional_strain'][-1]):.2E}, "
        outstr += f"curvature strain={np.max(outputs['curvature_strain'][-1]):.2E}, "
        outstr += f"frame twist={outputs['frame_twist'][-1]}\n"
        if len(wp_base_curves)>0:
            for i, (l, msc, jcs) in enumerate(zip(outputs['WP_length'][-1], outputs['WP_msc'][-1], outputs['WP_max_curvature'][-1])):
                outstr += f"WP_{i:d} length={l:.5E}, msc={msc:.5E}, max(c,0)^2={jcs:.5E}\n"
                outstr += f"WP max Z constraint={J_maxZ_wp.J():.5E}\n"
        outstr += f"\n"
                
        # Evaluate Jacobian - this is some magic math copied from Rogerio's code
        prob_dJ = prob_jacobian.jac(Jplasma.x)[0] # finite differences
        coils_dJ = Jcoils.dJ() # Analytical
        outputs['dJplasma'].append(prob_dJ)
        outputs['dJcoils'].append(coils_dJ)
    
        assert square_flux.definition == "local" #??
    
        # Evaluate how Jcoil varies w.r.t the surface dofs
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) \
              + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        grad_with_respect_to_coils = coils_objective_weight.value * coils_dJ
        mixed_dJ = Derivative({surf: deriv})(surf)
        
        ## Put both gradients together
        grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight.value * mixed_dJ

    # Print output string in log
    if verbose:
        log_print(outstr)

    # Save pickle every 10 iterations
    if np.mod(info['Nfeval'],10)==1 and comm_world.rank==0:
        with open(os.path.join(this_path, 'outputs.pckl'), 'wb') as f:
            pickle.dump( outputs, f )
    
    grad = np.concatenate((grad_with_respect_to_coils,grad_with_respect_to_surface))
    outputs['dJ'].append(grad)

    return J, grad

# For the first optimization stage, we only optimize the coils to match the surface initial guess.
Jcoils.fix_all()

# Unfix IL geometry
il_base_curve = il_curve
while hasattr(il_base_curve, 'curve'):
    il_base_curve = il_base_curve.curve
il_base_curve.name = 'IL_base_curve'

if inputs['cnt_coils']['dofs']['IL_geometry_free']:
    for ii in range(inputs['cnt_coils']['dofs']['IL_order']+1):
        il_base_curve.unfix(f'xc({ii})')
        if ii>0:
            il_base_curve.unfix(f'ys({ii})')
            il_base_curve.unfix(f'zs({ii})')

# Unfix PF current
if inputs['cnt_coils']['dofs']['PF_current_free']:
    pf_base_current.unfix_all()

# Unfix WP geometry
for c in wp_base_curves:
    for dofname in inputs['wp_coils']['dofs']['name']:
        c.unfix(dofname)

# Unfix WP current
for c in wp_base_coils:
    c.current.unfix_all()

# Define VMEC dofs
vmec.fix_all()
vmec.boundary.fixed_range(
    0, inputs['vmec']['dofs']['mpol'], 
    -inputs['vmec']['dofs']['ntor'], inputs['vmec']['dofs']['ntor'], 
    fixed=False
)

if not inputs['cnt_coils']['dofs']['R00_free']:
    vmec.boundary.fix('rc(0,0)')

if inputs['winding']['il_tor_weight'].value==0 and inputs['winding']['il_bincurv_weight'].value==0 and inputs['winding']['il_twist_weight']==0:
    rotation.fix_all()
else:
    rotation.unfix_all()


# Save initial degrees of freedom
log_print('The initial coils degrees of freedom are:\n')
if comm_world.rank == 0: 
    for name, dof in zip(Jcoils.dof_names, Jcoils.x):
        log_print(f"{name}={dof:.2e}\n")
    log_print("\n")

log_print('The initial surf degrees of freedom are:\n')
if comm_world.rank == 0: 
    for name, dof in zip(vmec.boundary.dof_names, vmec.boundary.x):
        log_print(f"{name}={dof:.2e}\n")
    log_print("\n")


dofs = np.concatenate((Jcoils.x, vmec.x))
x0 = np.copy(dofs) # Make a copy of dofs for weight iterations
ndof_vmec = int(len(vmec.boundary.x))

# Coil weight iteration
if inputs['numerics']['number_weight_iteration_stage_II']>0:
    satisfied = False
    counter = 0
    factor = inputs['numerics']['weight_iteration_factor']
    margin = inputs['numerics']['weight_margin']
    
    # Print in log
    log_print("Starting iteration on penalty weights... \n\n")
    while not satisfied and counter<inputs['numerics']['number_weight_iteration_stage_II']:
        counter += 1
        log_print(f"Weight iteration #{counter}...\n")
        
        # Re-initialize dofs
        set_dofs( x0 )
        dofs = np.concatenate((Jcoils.x, vmec.x))
  
        # Run quick stage II
        options={'maxiter': inputs['numerics']['MAXITER_weight_iteration_stage_II'], 'maxcor': 300}
        res = minimize(fun_coils, dofs[:-ndof_vmec], jac=True, method='L-BFGS-B', args=({'Nfeval': 0}, False), options=options, tol=1e-12)
        
    
        # Set satisfied to True; only False if one constraint is not satisfied
        satisfied = True
        
        # Evaluate targets and modify weights
        if inputs['CS_WEIGHT_iteration']:
            csdist = Jcsdist.shortest_distance()
            if csdist < (1-margin) * inputs['CS_THRESHOLD']:
                inputs['CS_WEIGHT'] *= factor
                satisfied = False
                log_print(f'CS shortest distance is {csdist:.2E}. Increasing weight...\n')

        if inputs['CC_WEIGHT_iteration']:
            ccdist = Jccdist.shortest_distance()
            if ccdist < (1-margin) * inputs['CC_THRESHOLD']:
                inputs['CS_WEIGHT'] *= factor
                satisfied = False
                log_print(f'CS shortest distance is {ccdist:.2E}. Increasing weight...\n')

        
        if inputs['cnt_coils']['target']['IL_length_weight_iteration']:
            ll = il_length.J()
            if ll > (1+margin) * il_length_target:
                il_length_weight *= facotr
                satisfied = False
                log_print(f'IL length is {ll:.2E}. Increasing weight...\n')

        if inputs['cnt_coils']['target']['IL_msc_weight_iteration']:
            msc = il_msc.J()
            if msc > (1+margin) * il_msc_threshold:
                il_msc_weight *= factor
                satisfied = False
                log_print(f'IL mean sqaure curvature is {msc:.2E}. Increasing weight...\n')


        if inputs['cnt_coils']['target']['IL_maxc_weight_iteration']:
            maxc = np.max(il_curve.kappa())
            if maxc > (1+margin) * il_curvature_threshold:
                il_curvature_weight *= factor
                satisfied = False
                log_print(f'IL max curvature is {maxc:.2E}. Increasing weight...\n')


        if inputs['cnt_coils']['target']['IL_maxR_weight_iteration']:
            g = il_curve.gamma()
            maxr = np.max(np.sqrt(g[:,1]**2 + g[:,2]**2))
            if maxr > (1+margin) * il_curveR_threshold:
                il_curveR_weight *= factor
                satisfied = False
                log_print(f'IL max R is {maxr:.2E}. Increasing weight...\n')
                
        if inputs['cnt_coils']['target']['IL_maxZ_weight_iteration']:
            g = il_curve.gamma()
            maxz = np.max(np.abs(g[:,0]))
            if maxz > (1+margin) * il_curveZ_threshold:
                il_curveZ_weight *= factor
                satisfied = False
                log_print(f'IL max Z is {maxz:.2E}. Increasing weight...\n')

        if inputs['cnt_coils']['target']['IL_vessel_weight_iteration']:
            min_coil_vessel_distance = vpenalty.minimum_distances()[0]
            if min_coil_vessel_distance < (1-margin) * il_vessel_threshold:
                il_vessel_weight *= factor
                satisfied = False
                log_print(f'IL min distance to vessel is {min_coil_vessel_distance:.2E}. Increasing weight...\n')

        if inputs['winding']['il_tor_weight_iteration']:
            torsional_strain = fc.torsional_strain()
            if torsional_strain > (1+margin)*inputs['winding']['tor_threshold']:
                il_tor_weight *= factor
                satisfied = False
                log_print(f'Torsional strain is {torsional_strain:.2E}. Increasing weight...\n')

        if inputs['winding']['il_bincurv_weight_iteration']:
            binormal_curvature_strain = fc.binormal_curvature_strain()
            if binormal_curvature_strain > (1+margin) * inputs['winding']['cur_threshold']:
                il_bincurv_weight *= factor
                satisfied = False
                log_print(f'Binormal curvature strain is {binormal_curvature_strain:.2E}. Increasing weight...\n')


        if inputs['winding']['il_twist_weight_iteration']:
            tw = twist.J()
            if tw > (1+margin) * inputs['winding']['il_twist_max']:
                il_twist_weight *= factor
                satisfied = False
                log_print(f'Twist penalty is {tw:.2E}. Increasing weight...\n')


        if satisfied:
            log_print('All penalties are satisfied! Saving weights...\n')

            if comm_world.rank == 0: 
                with open(os.path.join(this_path, 'input_updated_weights.pckl'), 'wb') as f:
                    pickle.dump(inputs, f)


        log_print('\n')
        

# Run the stage II optimization
if inputs['numerics']['MAXITER_stage_2'] > 0:
    log_print('Starting stage II optimization...')
    
    # Run minimization
    options={'maxiter': inputs['numerics']['MAXITER_stage_2'], 'maxcor': 300}
    res = minimize(fun_coils, dofs[:-ndof_vmec], jac=True, method='L-BFGS-B', args=({'Nfeval': 0}), options=options, tol=1e-12)

    if comm_world.rank==0:
        coils_to_vtk( full_coils, os.path.join(coils_results_path, "coils_post_stage_2") )
        surf_to_vtk( os.path.join(coils_results_path, "surf_post_stage_2"), bs, surf )
        bs.save( os.path.join(coils_results_path, "bs_post_stage_2.json") )
        bs_wp.save( os.path.join(coils_results_path, "bs_wp_post_stage_2.json") )
        fc.save( os.path.join(coils_results_path, "hts_frame_post_stage_2.json") )


# =================================================================================================
# FULL COMBINED OPTIMIZATION
# --------------------------
# We now construct the full, combined optimization problem. We define a new optimization function, that
# encompasses the coils objective, some plasma objectives, and the quadratic flux across the plasma boundary.

if inputs['numerics']['MAXITER_single_stage'] == 0:
    sys.exit()


# Write initial target values
log_print("=====================================================================\n Starting single stage optimization ...\n")
log_print(f"Aspect ratio before optimization: {vmec.aspect()}\n")
log_print(f"Mean iota before optimization: {vmec.mean_iota()}\n")
log_print(f"Quasisymmetry objective before optimization: {qs.total()}\n")
log_print(f"Magnetic well before optimization: {vmec.vacuum_well()}\n")
log_print(f"Squared flux before optimization: {square_flux.J()}\n")
log_print(f"Performing combined optimization with {inputs['numerics']['MAXITER_stage_2']} iterations\n")
log_print("\n")

# Define taylor test
dofs_coils = Jcoils.x
dofs_plasma = vmec.x

dofs = np.copy(np.concatenate((Jcoils.x,vmec.x)))
x0 = np.copy(dofs) # Save dofs for weight iterations


# Print initial degrees of freedom
log_print('The initial coils degrees of freedom are:\n')
if comm_world.rank == 0: 
    for name, dof in zip(Jcoils.dof_names, Jcoils.x):
        log_print(f"{name}={dof:.2e}\n")
    log_print("\n")
    for name, dof in zip(vmec.dof_names, vmec.x):
        log_print(f"{name}={dof:.2e}\n")
    log_print("\n")


# Iterate on weights
if inputs['numerics']['number_weight_iteration_single_stage']>0:
    satisfied = False
    counter = 0
    factor = inputs['numerics']['weight_iteration_factor']
    margin = inputs['numerics']['weight_margin']
    
    # Print in log
    log_print("Starting iteration on penalty weights... \n\n")
    while not satisfied and counter<inputs['numerics']['number_weight_iteration_single_stage']:
        counter += 1
        log_print(f"Weight iteration #{counter}...\n")
        
        # Re-initialize dofs
        set_dofs( x0 )
        dofs = np.concatenate((Jcoils.x, vmec.x))
  
        # Run quick stage II
        options={'maxiter': inputs['numerics']['MAXITER_weight_iteration_single_stage'], 'maxcor': 300}
        res = minimize(fun_coils, dofs[:-ndof_vmec], jac=True, method='L-BFGS-B', args=({'Nfeval': 0}, False), options=options, tol=1e-12)
        
        # Set satisfied to True; only False if one constraint is not satisfied
        satisfied = True

        if inputs['vmec']['target']['aspect_ratio_weight_iteration']:
            a = vmec.aspect()
            if a > inputs['vmec']['target']['aspect_ratio'] * (1+margin):
                inputs['vmec']['target']['aspect_ratio_weight'] *= factor
                satisfied = False
                log_print(f"Aspect ratio is {a:.2E}, increasing weights...\n")

        
        if inputs['vmec']['target']['iota_weight_iteration']:
            mean_iota = vmec.mean_iota()
            if mean_iota < inputs['vmec']['target']['iota'] * (margin+1):
                inputs['vmec']['target']['iota_weight'] *= factor
                satisfied = False
                log_print(f"Mean iota is {mean_iota:.2E}, increasing weights...\n")


        if inputs['vmec']['target']['volume_weight_iteration']:
            volume = vmec.volume()
            if volume < inputs['vmec']['target']['volume'] * (1+margin):
                inputs['vmec']['target']['volume_weight'] *= factor
                satisfied = False
                log_print(f"Volume is {volume:.2E}, increasing weights...\n")
                
        if satisfied:
            log_print('All penalties are satisfied! Saving weights...\n')

            if comm_world.rank == 0: 
                with open(os.path.join(this_path, 'input_updated_weights_2.pckl'), 'wb') as f:
                    pickle.dump(inputs, f)


        log_print('\n')                


myeps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
if inputs['numerics']['taylor_test']:
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    h1 = h[:-ndof_vmec]
    h2 = h[-ndof_vmec:]
    def taylor_test(f, x, h):
        out = np.zeros((len(myeps),))
        f0, df0 = f(x, info={'Nfeval':0, 'grad':True})
        df0 = df0.reshape(h.shape)
        for ii, eps in enumerate(myeps):
            f1, _ = f(x+eps*h, info={'Nfeval':0, 'grad':False})
            f2, _ = f(x-eps*h, info={'Nfeval':0, 'grad':False})
            out[ii] = (f1-f2) / (2*eps) - sum(df0*h)
        return out
    
    def fun_plasma(x0, prob_jacobian, info):
        info['Nfeval'] += 1
        Jplasma.x = x0
        if info['grad']:
            return Jplasma.J(), prob_jacobian.jac(x0)
        else:
            return Jplasma.J(), 0
    
# Prepare output
outputs['result'] = None

# Define finite diff for vmec objectives and run minimizer
diff_method = inputs['numerics']['fndiff_method']
finite_difference_abs_step = inputs['numerics']['finite_difference_abs_step'] 
finite_difference_rel_step = inputs['numerics']['finite_difference_rel_step'] 
with MPIFiniteDifference(Jplasma.J, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
    if mpi.proc0_world:
        # Taylor test - coils            
        fpl = lambda x, info: fun_plasma(x, prob_jacobian, info)
        outputs['taylor_test'] = dict()
        outputs['taylor_test']['initial'] = dict()
        outputs['taylor_test']['eps'] = myeps

        if inputs['numerics']['taylor_test']:
            log_print('-----------------------------------------------------------------\n')
            log_print('                              INITIAL TAYLOR TEST \n')
            log_print('Running initial Taylor test for coils...\n')
            outputs['taylor_test']['initial']['Jcoils'] = taylor_test(fun_coils, dofs_coils, h1)
            log_print('Running initial Taylor test for plasma...\n')
            outputs['taylor_test']['initial']['Jplasma'] = taylor_test(fpl, dofs_plasma, h2)
            log_print('Running initial Taylor test for full objective...\n')
            outputs['taylor_test']['initial']['Jtotal'] = taylor_test(fun, dofs, h)
            
        # -------------------------------------------------------------------------------------
        outputs['result'] = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': inputs['numerics']['MAXITER_single_stage']}, tol=1e-12)    

        res = outputs['result']
        log_print(f"Number of iterations: {res.nit}\n")
        log_print(f"Number of function evaluations: {res.nfev}\n")
        log_print(f"Optimization status: {res.status}\n")
        log_print(f"Final function value: {res.fun}\n")
        log_print(f"Final gradient: {res.jac}\n" )
        log_print(f"\n")
        try:
            log_print(f"Aspect ratio after optimization: {vmec.aspect()}\n")
            log_print(f"Mean iota after optimization: {vmec.mean_iota()}\n")
            log_print(f"Quasisymmetry objective after optimization: {qs.total()}\n")
            log_print(f"Magnetic well after optimization: {vmec.vacuum_well()}\n")
        except BaseException as e:
            log_print(f"Could not print final aspect ratio, mean iota, QS, and vacuum well")
        log_print(f"Squared flux after optimization: {square_flux.J()}\n")
        
        for j, coil in enumerate(base_coils):
            log_print(f"Current for coil {j}: {coil.current.get_value()}\n")
        log_print("\n")
        
        # -------------------------------------------------------------------------------------
        # Taylor test
        if inputs['numerics']['taylor_test']:
            log_print('----------------------------------------------\n')
            log_print('                              FINAL TAYLOR TEST \n')
            outputs['taylor_test']['final'] = dict()
            log_print('Running final Taylor test for coils...')
            outputs['taylor_test']['final']['Jcoils'] = taylor_test(fun_coils, dofs_coils, h1)
            log_print('Running final Taylor test for plasma...')
            outputs['taylor_test']['final']['Jplasma'] = taylor_test(fpl, dofs_plasma, h2)
            log_print('Running final Taylor test for full objective...')
            outputs['taylor_test']['final']['Jtotal'] = taylor_test(fun, dofs, h)

mpi.comm_world.Bcast(dofs, root=0)   
        
# Save output
if comm_world.rank==0:    
    coils_to_vtk( full_coils, os.path.join(coils_results_path, "coils_output") )
    surf_to_vtk( os.path.join(coils_results_path, "surf_output"), bs, surf )

    with open(os.path.join(this_path, 'outputs.pckl'), 'wb') as f:
        pickle.dump( outputs, f )

    bs.save( os.path.join(coils_results_path, "bs_output.json") )
    bs_wp.save( os.path.join(coils_results_path, "bs_wp_output.json") )
    vmec.write_input(os.path.join(this_path, f'input.final'))
    fc.save( os.path.join(coils_results_path, "hts_frame_final.json") )

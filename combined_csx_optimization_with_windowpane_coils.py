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
"""
# Import and metadata
# -------------------
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
from simsopt._core import Optimizable
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries, apply_symmetries_to_curves
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem, Weight
from simsopt.geo import CurveLength, CurveCurveDistance, MeanSquaredCurvature,  LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves, create_equally_spaced_windowpane_curves, CurveSurfaceDistance
from simsopt.geo.windowpanecurve import WindowpaneCurveXYZFourier
from simsopt.field.coilobjective import CurrentPenalty
from simsopt.field.coil import apply_symmetries_to_currents, ScaledCurrent
from simsopt.field.coil import Coil
from set_default_values import set_default
from simsopt._core.util import ObjectiveFailure
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve

# Setup MPI
comm = MPI.COMM_WORLD
mpi = MpiPartition()

# Define new screen output...
def pprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)

# Path
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
print('parent_path: ',parent_path)

# Create output directories
date = datetime.datetime
hms = f'{date.now().year}{date.now().month}{date.now().day}' +\
      f'_{date.now().hour}{date.now().minute}{date.now().second}'

std = importlib.import_module(sys.argv[1], package=None)
inputs = std.inputs
set_default(inputs)

if 'directory' in inputs.keys():
    dir_name = inputs['directory']
else:
    dir_name = 'runs/' + date.now().isoformat(timespec='seconds') + '/'

if comm.rank == 0: 
    os.makedirs(dir_name, exist_ok=True)

this_path = os.path.join(parent_path, dir_name)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm.rank == 0: 
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)

# Save input
if comm.rank == 0: 
    with open(os.path.join(this_path, 'input.pckl'), 'wb') as f:
        pickle.dump(inputs, f)

def print_dict_recursive(file, d, order=0, k=None):
    if type(d) is dict:
        for k, i in d.items():
            if type(i) is dict:
                for l in range(order):
                    file.write('\n')
                for l in range(order):
                    file.write('#')
                if order>0:
                    file.write(' ')
                file.write(f'{k}\n')
            print_dict_recursive(file, i, order=order+1, k=k)
            file.write(f' \n')
    else:
        for l in range(order-1):
            file.write('')
        file.write(f'{k} = {d}')

if comm.rank == 0: 
    with open(os.path.join(this_path, 'input.txt'), 'w') as f:
        print_dict_recursive(f, inputs)


# =================================================================================================
# CREATE INITIAL COILS AND SURFACE
# --------------------------------

outstr  = "=============== COMBINED OPTIMIZATION OF CSX ===============\n"
outstr += f"THIS IS CSX COMBINED OPTIMIZATION\n"
outstr += f"Date = {date.date(date.now()).isoformat()} at {date.now().strftime('%Hh%M')}\n"
outstr += "\n\n"
if comm.rank == 0: 
    with open(os.path.join(this_path,'log.txt'), 'w') as f:
        f.write(outstr)


# Load Vmec object, extract the boundary
vmec = Vmec(
    os.path.join( parent_path, inputs['vmec']['filename'] ),
    mpi=mpi, 
    verbose=inputs['vmec']['verbose'], 
    nphi=inputs['vmec']['nphi'], 
    ntheta=inputs['vmec']['nphi']
)
vmec.indata.mpol = inputs['vmec']['internal_mpol'] 
vmec.indata.ntor = inputs['vmec']['internal_ntor'] 

# Save initial vmec
vmec.write_input(os.path.join(this_path, f'input.initial'))

surf = vmec.boundary

#Load CNT initial coils. Extract the base curves and currents.
cnt_initial_coils = load( inputs['cnt_coils']['geometry']['filename'] )
il_base_coil = cnt_initial_coils.coils[0]
il_coils = cnt_initial_coils.coils[0:2]
pf_base_coil = cnt_initial_coils.coils[2]
pf_coils = cnt_initial_coils.coils[2:4]
del(cnt_initial_coils)

il_base_curve = il_coils[0].curve
while hasattr(il_base_curve, 'curve'):
    il_base_curve = il_base_curve.curve
il_base_curve.name = 'IL_base_curve'

il_base_current = il_coils[0].current
il_base_current.name = 'IL_base_current'

pf_base_curve = pf_coils[0].curve
while hasattr(pf_base_curve, 'curve'):
    pf_base_curve = pf_base_curve.curve
pf_base_curve.name = 'PF_base_curve'

pf_base_current = pf_coils[0].current
pf_base_current.name = 'PF_base_current'


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
    bs = load( inputs['wp_coils']['geometry']['filename'] )
    ncpr = inputs['wp_coils']['geometry']['ncoil_per_row']
    nr = len(inputs['wp_coils']['geometry']['Z0'])
    wp_base_coils = bs.coils[4:4+ncpr*nr]
    wp_coils = bs.coils[4:]
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
bs.set_points( surf.gamma().reshape((-1,3)) )

# Save initial coils and surface
coils_to_vtk( full_coils, os.path.join(coils_results_path, "initial_coils") )
surf_to_vtk( os.path.join(coils_results_path, "initial_surface"), bs, surf )
bs.save( os.path.join(coils_results_path, "bs_initial.json") )



# =================================================================================================
# RUN FIRST STAGE TWO OPTIMIZATION
# --------------------------------
# We begin with a stage two optimization to get the coils as close as possible to the VMEC 
# boundary. Here, we only include coils penalty function and attempt at minimizing the quadratic
# flux across VMEC boundary.

Jf = SquaredFlux(surf, bs, definition="local")
JF = Jf

def add_penalty( JF, newJ, w ):
    if w.value>0:
        JF += w * newJ
    return JF

# IL penalties
il_length = CurveLength( il_base_curve )
JF = add_penalty( JF,
    QuadraticPenalty( il_length, inputs['cnt_coils']['target']['IL_length'], f=inputs['cnt_coils']['target']['IL_length_constraint_type'] ), 
    inputs['cnt_coils']['target']['IL_length_weight'] 
)

il_curvature = LpCurveCurvature(il_base_curve, 2, inputs['cnt_coils']['target']['IL_maxc_threshold'])
JF = add_penalty( JF,
    il_curvature,
    inputs['cnt_coils']['target']['IL_maxc_weight']
)

il_msc = MeanSquaredCurvature( il_base_curve )
JF = add_penalty( JF,
    QuadraticPenalty(il_msc, inputs['cnt_coils']['target']['IL_msc_threshold'], f='max'),
    inputs['cnt_coils']['target']['IL_msc_weight'] 
)

# WP penalties
if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
    wp_lengths = [CurveLength( c ) for c in wp_base_curves]
    JF = add_penalty( JF,
        sum([QuadraticPenalty( wpl, inputs['wp_coils']['target']['length'], f=inputs['wp_coils']['target']['length_constraint_type ']) for wpl in wp_lengths]),
        inputs['wp_coils']['target']['length_weight']
    )
    
    wp_curvatures = [LpCurveCurvature(c, 2, inputs['wp_coils']['target']['maxc_threshold']) for c in wp_base_curves]
    JF = add_penalty( JF,
        sum(wp_curvatures),
        inputs['wp_coils']['target']['maxc_weight']
    )

    wp_msc = [MeanSquaredCurvature(c) for c in wp_base_curves]
    JF = add_penalty( JF,
        sum([QuadraticPenalty(msc, inputs['wp_coils']['target']['msc_threshold'], f='max') for msc in wp_msc]),
        inputs['wp_coils']['target']['msc_weight']
    )

Jccdist = CurveCurveDistance(full_curves, inputs['CC_THRESHOLD'], num_basecurves=len(full_curves))
JF = add_penalty( JF,
    Jccdist,
    inputs['CC_WEIGHT']
)
Jcsdist = CurveSurfaceDistance(base_curves, surf, inputs['CS_THRESHOLD'])
JF = add_penalty( JF,
    Jcsdist,
    inputs['CS_WEIGHT']
)

coils_objective = (inputs['coils_objective_weight'].value>0)
def fun_coils(dofs, info):
    info['Nfeval'] += 1
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        nphi_VMEC = vmec.boundary.quadpoints_phi.size
        ntheta_VMEC = vmec.boundary.quadpoints_theta.size
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f"IL length={il_length.J():.2f},  IL ∫ϰ²/L={il_msc.J():.2f},  IL ∫max(ϰ-ϰ0,0)^2={il_curvature.J():.2f}\n"
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
            for i, (l, msc, jcs) in enumerate(zip(wp_lengths, wp_msc, wp_curvatures)):
                outstr += f"WP_{i:02.0f} length={l.J():.2f},  WP_{i:02.0f} ∫ϰ²/L={msc.J():.2f},  WP_{i:02.0f} ∫max(ϰ-ϰ0,0)^2={jcs.J():.2f}\n" 
            outstr += f"\n"

        with open(os.path.join(this_path,'log.txt'), 'a') as f:
            f.write(outstr)

    return J, grad


# For the first optimization stage, we only optimize the coils to match the surface initial guess.
JF.fix_all()

# Unfix IL geometry
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
    for xyz in ['x','y','z']:
        c.unfix(f'{xyz}0')
    for ypr in ['yaw','pitch','roll']:
        c.unfix(f'{ypr}')
    for ii in range(1,inputs['wp_coils']['dofs']['order']+1):
        c.unfix(f'xc({ii})')
        c.unfix(f'xs({ii})')
        c.unfix(f'zc({ii})')
        c.unfix(f'zs({ii})')


# Unfix WP current
for c in wp_base_currents:
    c.unfix_all()

# Save initial degrees of freedom
print(JF.dof_names)
dofs = JF.x
if comm.rank == 0: 
    with open(os.path.join(this_path,'log.txt'), 'a') as f:
        for name, dof in zip(JF.dof_names, JF.x):
            f.write(f"{name}={dof:.2e},  ")
        f.write("\n")

if inputs['numerics']['MAXITER_stage_1'] > 0 and coils_objective:
    # Run minimization
    options={'maxiter': inputs['numerics']['MAXITER_stage_1'], 'maxcor': 300}
    res = minimize(fun_coils, dofs, jac=True, method='L-BFGS-B', args=({'Nfeval': 0}), options=options, tol=1e-12)
    
    # Save coils and surface post stage-two
    coils_to_vtk( full_coils, os.path.join(coils_results_path, "coils_post_stage_2") )
    surf_to_vtk( os.path.join(coils_results_path, "surf_post_stage_2"), bs, surf )
    bs.save( os.path.join(coils_results_path, "bs_post_stage_2.json") )





# =================================================================================================
# FULL COMBINED APPROACH
# ----------------------

if inputs['numerics']['MAXITER_stage_2'] == 0:
    sys.exit()

# We now include both the coil penalty and the plasma target functions
outputs = dict()
outputs['J'] = []
outputs['Jplasma'] = []
outputs['Jcoils'] = []
outputs['iota_axis'] = []
outputs['iota_edge'] = []
outputs['mean_iota'] = []
outputs['aspect'] = []
outputs['QSresiduals'] = []
outputs['QuadFlux'] = []
outputs['BdotN'] = []
outputs['min_CS'] = []
outputs['min_CC'] = []
outputs['IL_length'] = []
outputs['WP_length'] = []
outputs['IL_msc'] = []
outputs['WP_msc'] = []
outputs['IL_max_curvature'] = []
outputs['WP_max_curvature'] = []
outputs['vmec'] = dict()
outputs['vmec']['fsqr'] = []
outputs['vmec']['fsqz'] = []
outputs['vmec']['fsql'] = []

qs = QuasisymmetryRatioResidual(
        vmec, inputs['vmec']['target']['qa_surface'], helicity_m=1, helicity_n=0, 
        ntheta=inputs['vmec']['target']['qa_ntheta'], nphi=inputs['vmec']['target']['qa_nphi']
)

class remake_iota(Optimizable):
    def __init__(self, vmec):
        self.vmec = vmec
        super().__init__(depends_on=[vmec])
    def J(self):
        "returns the value of the quantity"
        try:
            return self.vmec.mean_iota()
        except ObjectiveFailure: 
            with open(os.path.join(this_path, 'log.txt'), 'a') as f:
                f.write(f"Error evaluating iota! ")
            return np.nan

class remake_aspect(Optimizable):
    def __init__(self, vmec):
        self.vmec = vmec
        super().__init__(depends_on=[vmec])
    def J(self):
        "returns value of quantity"
        try: 
            return self.vmec.aspect()
        except ObjectiveFailure: 
            with open(os.path.join(this_path, 'log.txt'), 'a') as f:
                f.write(f"Error evaluating aspect ratio! ")
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

if coils_objective:
    J_iota = inputs['vmec']['target']['iota_weight'] * QuadraticPenalty(remake_iota(vmec), inputs['vmec']['target']['iota'], inputs['vmec']['target']['iota_constraint_type'])
    J_aspect = inputs['vmec']['target']['aspect_ratio_weight'] * QuadraticPenalty(remake_aspect(vmec), inputs['vmec']['target']['aspect_ratio'], inputs['vmec']['target']['aspect_ratio_constraint_type'])
    J_qs = QuadraticPenalty(quasisymmetry(qs), 0, 'identity')
    JF_2 = J_iota + J_aspect+ J_qs
else:
    # nonlinear constraints for iota and aspect ratio
    if inputs['vmec']['target']['aspect_ratio_constraint_type']=='identity':
        aspect_tuple = (vmec.aspect, inputs['vmec']['target']['aspect_ratio'], inputs['vmec']['target']['aspect_ratio'])
    elif inputs['vmec']['target']['aspect_ratio_constraint_type']=='max':
        aspect_tuple = (vmec.aspect, 0, inputs['vmec']['target']['aspect_ratio'])
    elif inputs['vmec']['target']['aspect_ratio_constraint_type']=='min':
        aspect_tuple = (vmec.aspect, inputs['vmec']['target']['aspect_ratio'], np.inf)
    
    if inputs['vmec']['target']['iota_constraint_type']=='identity':
        iota_tuple = (vmec.mean_iota, inputs['vmec']['target']['iota'], inputs['vmec']['target']['iota'])
    elif inputs['vmec']['target']['iota_constraint_type']=='max':
        iota_tuple = (vmec.mean_iota, -np.inf, inputs['vmec']['target']['iota'])
    elif inputs['vmec']['target']['iota_constraint_type']=='min':
        iota_tuple = (vmec.mean_iota, inputs['vmec']['target']['iota'], np.inf)

    tuples_nlc = [aspect_tuple, iota_tuple]
    
    # Define problem
    prob = ConstrainedProblem(qs.total, tuples_nlc=tuples_nlc)

# Define target function
JACOBIAN_THRESHOLD = inputs['numerics']['JACOBIAN_THRESHOLD']

def fun(dofs, prob_jacobian=None, info={'Nfeval':0}):
    info['Nfeval'] += 1
    coils_objective_weight = inputs['coils_objective_weight']
    
    # Keep all vmec files
    # vmec.files_to_delete = []

    # Separate coils dofs from surface dofs
    JF.x = dofs[:-ndof_vmec]
    JF_2.x = dofs[-ndof_vmec:]

    # Update surface
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # Evaluate target function
    os.chdir(vmec_results_path)
    J_stage_1 = JF_2.J()
    J_stage_2 = coils_objective_weight.value * JF.J()
    J = J_stage_1 + J_stage_2

    outputs['J'].append(float(J))
    outputs['Jplasma'].append(float(J_stage_1))
    outputs['Jcoils'].append(float(J_stage_2))
    outputs['vmec']['fsqr'].append(vmec.wout.fsqr)
    outputs['vmec']['fsqz'].append(vmec.wout.fsqz)
    outputs['vmec']['fsql'].append(vmec.wout.fsql)
        
    if J > inputs['numerics']['JACOBIAN_THRESHOLD'] or np.isnan(J):
        if comm.rank==0:
            with open(os.path.join(this_path, 'log.txt'), 'a') as f:
                f.write(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}\n")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * ndof_vmec
        grad_with_respect_to_coils = [0] * len(JF.x)

        outputs['iota_axis'].append(np.nan)
        outputs['iota_edge'].append(np.nan)
        outputs['mean_iota'].append(np.nan)
        outputs['aspect'].append(np.nan)
        outputs['QSresiduals'].append(np.nan)
        outputs['QuadFlux'].append(np.nan)
        outputs['BdotN'].append(np.nan)
        outputs['min_CS'].append(np.nan)
        outputs['min_CC'].append(np.nan)
        outputs['IL_length'].append(np.nan)
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0: 
            outputs['WP_length'].append([np.nan for l in wp_lengths])
            outputs['WP_msc'].append([np.nan for msc in wp_msc])
            outputs['WP_max_curvature'].append([np.nan for c in wp_curvatures])
        outputs['IL_msc'].append(np.nan)
        outputs['IL_max_curvature'].append(np.nan)

        outstr = f"STEP {info['Nfeval']:03.0f}. J_stage_1 = {J_stage_1}, J_stage_2 = {J_stage_2}\n"
        outstr += "VMEC FAILED"
        grad_with_respect_to_surface = [0] * ndof_vmec
        grad_with_respect_to_coils = [0] * len(JF.x)
    
    else:
        # Evaluate important metrics    
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        nphi_VMEC = surf.quadpoints_phi.size
        ntheta_VMEC = surf.quadpoints_theta.size
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        B_n = np.sum(Bcoil*unitn, axis=2)     # This is B.n/|n|
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2) # This is |B|
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)

        # Save in output arrays
        outputs['iota_axis'].append(float(vmec.iota_axis()))
        outputs['iota_edge'].append(float(vmec.iota_edge()))
        outputs['mean_iota'].append(float(vmec.mean_iota()))
        outputs['aspect'].append(float(vmec.aspect()))
        outputs['QSresiduals'].append(np.array(qs.residuals()))
        outputs['QuadFlux'].append(float(Jf.J()))
        outputs['BdotN'].append(np.array(B_n))
        outputs['min_CS'].append(float(Jcsdist.shortest_distance()))
        outputs['min_CC'].append(float(Jccdist.shortest_distance()))
        outputs['IL_length'].append(float(il_length.J()))
        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0: 
            outputs['WP_length'].append([float(l.J()) for l in wp_lengths])
            outputs['WP_msc'].append([float(msc.J()) for msc in wp_msc])
            outputs['WP_max_curvature'].append([float(c.J()) for c in wp_curvatures])
        outputs['IL_msc'].append(float(il_msc.J()))
        outputs['IL_max_curvature'].append(float(il_curvature.J()))

   
        # Log output
        outstr = f"STEP {info['Nfeval']:03.0f}. J_stage_1 = {J_stage_1}, J_stage_2 = {J_stage_2}\n"
        outstr += f"aspect={outputs['aspect'][-1]}, iota_axis={outputs['iota_axis'][-1]}, iota_edge={outputs['iota_edge'][-1]}, iota={outputs['mean_iota'][-1]}\n"
        outstr += f"QS residuals={outputs['QSresiduals'][-1]}"
        outstr += f"Jf={outputs['QuadFlux'][-1]:.1e}, ⟨B·n⟩={np.mean(np.abs(B_n)):.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", C-C-Sep={outputs['min_CC'][-1]}"
        outstr += f", C-S-Sep={outputs['min_CS'][-1]}"
        outstr += f"IL length={outputs['IL_length'][-1]:.2f},  IL ∫ϰ²/L={outputs['IL_msc'][-1]:.2f},  IL ∫max(ϰ-ϰ0,0)^2={outputs['IL_max_curvature'][-1]:.2f}\n"

        if inputs['wp_coils']['geometry']['ncoil_per_row'] > 0:
            for i, (l, msc, jcs) in enumerate(zip(outputs['WP_length'][-1], outputs['WP_msc'][-1], outputs['WP_max_curvature'][-1])):
                outstr += f"WP_{i:02.0f} length={l:.2f}, ∫ϰ²/L={msc:.2f}, ∫max(ϰ-ϰ0,0)^2={jcs:.2f}\n" 
        outstr += f"\n"
                
        # Evaluate Jacobian - this is some magic math copied from Rogerio's code
        prob_dJ = prob_jacobian.jac(JF_2.x)[0] # finite differences
    
        coils_dJ = JF.dJ() # Analytical
    
        assert Jf.definition == "local"
    
        # Evaluate how Jcoil varies w.r.t the surface dofs
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) \
              + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        grad_with_respect_to_coils = coils_objective_weight.value * coils_dJ
        mixed_dJ = Derivative({surf: deriv})(surf)
        
        ## Put both gradients together
        grad_with_respect_to_surface = np.ravel(prob_dJ) \
        + coils_objective_weight.value * mixed_dJ


    if comm.rank==0:
        with open(os.path.join(this_path,'log.txt'), 'a') as f:
            f.write(outstr)
    
    #if np.mod(info['Nfeval'],5)==1 and comm.rank==0:
    with open(os.path.join(this_path, 'outputs.pckl'), 'wb') as f:
        pickle.dump( outputs, f )
    
    grad = np.concatenate((grad_with_respect_to_coils,grad_with_respect_to_surface))

    return J, grad



# Define VMEC dofs
vmec.fix_all()
vmec.boundary.fixed_range(
    0, inputs['vmec']['dofs']['mpol'], 
    -inputs['vmec']['dofs']['ntor'], inputs['vmec']['dofs']['ntor'], 
    fixed=False
)
vmec.boundary.fix('rc(0,0)') # We want to keep major radius fixed.

dofs = np.concatenate((JF.x, vmec.x))
ndof_vmec = int(len(vmec.boundary.x))

# Write initial target values
if comm.rank == 0: 
    with open(os.path.join(this_path,'log.txt'), 'a') as f:
        f.write(f"Aspect ratio before optimization: {vmec.aspect()}\n")
        f.write(f"Mean iota before optimization: {vmec.mean_iota()}\n")
        f.write(f"Quasisymmetry objective before optimization: {qs.total()}\n")
        f.write(f"Magnetic well before optimization: {vmec.vacuum_well()}\n")
        f.write(f"Squared flux before optimization: {Jf.J()}\n")
        f.write(f"Performing stage 2 optimization with {inputs['numerics']['MAXITER_stage_2']} iterations\n")
        f.write("\n")

# Define finite diff for vmec objectives and run minimizer
diff_method = inputs['numerics']['fndiff_method']
finite_difference_abs_step = inputs['numerics']['finite_difference_abs_step'] 
finite_difference_rel_step = inputs['numerics']['finite_difference_rel_step'] 

if comm.rank==0:
    with open(os.path.join(this_path,'simsopt_single_stage_metric.txt'), 'w') as f:
        f.write("Starting single stage optimization ...\n")

outputs['result'] = None

if not coils_objective:
    options = {'disp': True, 'ftol': 1e-7, 'maxiter': inputs['numerics']['MAXITER_stage_2']}
    constrained_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, options=options)
else:
    with MPIFiniteDifference(JF_2.J, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
        if mpi.proc0_world:
            outputs['result'] = minimize(
                    fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', 
                    options={'maxiter': inputs['numerics']['MAXITER_stage_2']}, tol=1e-15
                    )
            if comm.rank == 0:
                res = outputs['result']
                with open(os.path.join(this_path,'log.txt'), 'a') as f:
                    f.write(f"Number of iterations: {res.nit}\n")
                    f.write(f"Number of function evaluations: {res.nfev}\n")
                    f.write(f"Optimization status: {res.status}\n")
                    f.write(f"Final function value: {res.fun}\n")
                    f.write(f"Final gradient: {res.jac}\n" )
                    f.write(f"\n")
                    
                    f.write(f"Aspect ratio after optimization: {vmec.aspect()}\n")
                    f.write(f"Mean iota after optimization: {vmec.mean_iota()}\n")
                    f.write(f"Quasisymmetry objective after optimization: {qs.total()}\n")
                    f.write(f"Magnetic well after optimization: {vmec.vacuum_well()}\n")
                    f.write(f"Squared flux after optimization: {Jf.J()}\n")
            
                    for j, coil in enumerate(base_coils):
                        f.write(f"Current for coil {j}: {coil.current.get_value()}\n")
                        
                    f.write("\n")

        with open(os.path.join(this_path, 'outputs.pckl'), 'wb') as f:
            pickle.dump( outputs, f )
            
    # Save output
    coils_to_vtk( full_coils, os.path.join(coils_results_path, "coils_output") )
    surf_to_vtk( os.path.join(coils_results_path, "surf_output"), bs, surf )

bs.save( os.path.join(coils_results_path, "bs_output.json") )
vmec.write_input(os.path.join(this_path, f'input.final'))

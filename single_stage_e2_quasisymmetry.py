#!/usr/bin/env python
r"""
In this example we both a stage 1 and stage 2 optimization problems using the
single stage approach of R. Jorge et al in https://arxiv.org/abs/2302.10622
The objective function in this case is J = J_stage1 + coils_objective_weight*J_stage2
To accelerate convergence, a stage 2 optimization is done before the single stage one.
Rogerio Jorge, April 2023
"""
import os
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt._core.optimizable import load
from mpi4py import MPI
from pathlib import Path
from scipy.optimize import minimize
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import CurveLength, CurveCurveDistance, MeanSquaredCurvature,  LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves

# LOAD COILS
comm = MPI.COMM_WORLD
bs = load('flux_100_bs_cssc_cssc.json')
def pprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)



mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
print('parent_path: ',parent_path)


##########################################################################################
############## Input parameters
##########################################################################################
MAXITER_stage_2 = 10
MAXITER_single_stage = 100
max_mode = 3 

# Fourier modes of the boundary with m <= max_mode and |n| <= max_mode
# will be varied in the optimization. A larger range of modes are
# included in the VMEC and booz_xform calculations.
surf_input = 'input.CSSCscaled3'

aspect_ratio_target = 3.5
iota_target_value = -0.2
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = 34
ntheta_VMEC = 34
nmodes_coils = 7
coils_objective_weight = 1e-01
aspect_ratio_weight = 1e+1
iota_target_weight = 1e+2 
diff_method = "forward"
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 0
finite_difference_rel_step = 1e-5
JACOBIAN_THRESHOLD = 100
LENGTH_CON_WEIGHT = 0.1  # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-8  # Weight on the curve lengths in the objective function
CC_WEIGHT = 1e+0  # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-4  # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-4  # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9  # Weight for the arclength variation penalty in the objective function


##########################################################################################
##########################################################################################
import datetime
date = datetime.datetime
directory = f'{date.now().year}{date.now().month}{date.now().day}_{date.now().hour}{date.now().minute}{date.now().second}'
vmec_verbose = True
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm.rank == 0: 
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
#pprint(f' Using vmec input file {vmec_input_filename}')
print('parent_path: ',parent_path)
vmec = Vmec(parent_path+'/'+surf_input, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC)
surf = vmec.boundary

##########################################################################################
##########################################################################################
# Save initial surface and coil data

coils = bs.coils
currents = [coil.current for coil in coils]
currents[0].fix_all()
base_curves = [coil.curve for coil in coils]
for j, current in enumerate(currents):
    print(f"Current for coil {j}: {current.get_value()}")

##########################################################################################
##########################################################################################
Jf = SquaredFlux(surf, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(base_curves, CC_THRESHOLD, num_basecurves=len(base_curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE + J_MSC
##########################################################################################
pprint(f'  Starting optimization')
##########################################################################################
# Initial stage 2 optimization
##########################################################################################
## The function fun_coils defined below is used to only optimize the coils at the beginning
## and then optimize the coils and the surface together. This makes the overall optimization
## more efficient as the number of iterations needed to achieve a good solution is reduced.


def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        # BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
        print(outstr)
    return J, grad
##########################################################################################
##########################################################################################
## The function fun defined below is used to optimize the coils and the surface together.


def fun(dofs, prob_jacobian=None, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    vmec.files_to_delete = []
    if mpi.proc0_world:
        print(f"Done optimization with max_mode ={max_mode}. "
              f"Final vmec iteration = {vmec.iter}")
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    os.chdir(vmec_results_path)
    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    if J > JACOBIAN_THRESHOLD or np.isnan(J):
        pprint(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(JF.x)
    else:
        pprint(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        prob_dJ = prob_jacobian.jac(prob.x)
        ## Finite differences for the second-stage objective function
        coils_dJ = JF.dJ()
        ## Mixed term - derivative of squared flux with respect to the surface shape
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
        B_n = Bcoil_n
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)
        assert Jf.definition == "local"
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        mixed_dJ = Derivative({surf: deriv})(surf)
        ## Put both gradients together
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    return J, grad


##########################################################################################
#############################################################
## Perform optimization
#############################################################
#########################################################################################
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=0)
objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1, 1e+2), (vmec.mean_iota, iota_target_value, iota_target_weight)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
dofs = np.concatenate((JF.x, vmec.x))
bs.set_points(surf.gamma().reshape((-1, 3)))
Jf = SquaredFlux(surf, bs, definition="local")
pprint(f"Aspect ratio before optimization: {vmec.aspect()}")
pprint(f"Mean iota before optimization: {vmec.mean_iota()}")
pprint(f"Quasisymmetry objective before optimization: {qs.total()}")
pprint(f"Magnetic well before optimization: {vmec.vacuum_well()}")
pprint(f"Squared flux before optimization: {Jf.J()}")
pprint(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm.rank == 0:
    curves_to_vtk(base_curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
pprint(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
x0 = np.copy(np.concatenate((JF.x, vmec.x)))
dofs = np.concatenate((JF.x, vmec.x))
with MPIFiniteDifference(prob.objective, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
    if mpi.proc0_world:
        res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-15)
        print("Number of iterations:", res.nit)
        print("Number of function evaluations:", res.nfev)
        print("Optimization status:", res.status)
        print("Final function value:", res.fun)
        print("Final gradient:", res.jac)
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm.rank == 0:
    curves_to_vtk(base_curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
print('biot savart here')
vmec.write_input(os.path.join(this_path, f'input.final'))
pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
pprint(f"Magnetic well after optimization: {vmec.vacuum_well()}")
pprint(f"Squared flux after optimization: {Jf.J()}")
for j, current in enumerate(currents):
    print(f"Current for coil {j}: {current.get_value()}")
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
BdotN = np.mean(np.abs(BdotN_surf))
BdotNmax = np.max(np.abs(BdotN_surf))
outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
pprint(outstr)





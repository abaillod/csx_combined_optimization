#!/usr/bin/env python
import os
import time
import glob
import logging
import matplotlib
import warnings
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from simsopt.mhd import VirtualCasing
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable, make_optimizable
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                        LpCurveCurvature, ArclengthVariation, CurveSurfaceDistance)
from simsopt.objectives import SquaredFlux, LeastSquaresProblem, QuadraticPenalty
from simsopt._core.finite_difference import finite_difference_steps, FiniteDifference, MPIFiniteDifference
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
logger = logging.getLogger(__name__)
mpi = MpiPartition()


# Input parameters
only_grad_stage2 = False
QA_or_QHs = ['QH','QA']
derivative_algorithm = "centered"
rel_step = 0
abs_step = 1e-8
eps_array = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
LENGTHBOUND=20
LENGTH_CON_WEIGHT=1e-2
JACOBIAN_THRESHOLD=50
CURVATURE_THRESHOLD=5
CURVATURE_WEIGHT=1e-6
MSC_THRESHOLD=5
MSC_WEIGHT=1e-6
CC_THRESHOLD=0.1
CC_WEIGHT = 1e-3
CS_THRESHOLD=0.05
CS_WEIGHT = 1e-3
ARCLENGTH_WEIGHT = 1e-7
max_mode=1
coils_objective_weight=1
ncoils=3
R0=1
R1=0.5
order=2
nphi=30
ntheta=30
finite_beta = False
vc_src_nphi = nphi
OUT_DIR = f"output"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

if mpi.proc0_world: print(f"""################################################################################
### Performing a Taylor test ###################################################
################################################################################
""")
fig = plt.figure(figsize=(8, 6), dpi=200)
ax = plt.subplot(111)
for QA_or_QH in QA_or_QHs:
    if finite_beta: vmec_file = f'../input.precise{QA_or_QH}_FiniteBeta'
    else: vmec_file = f'../input.precise{QA_or_QH}'
    vmec = Vmec(vmec_file, nphi=nphi, ntheta=ntheta, mpi=mpi, verbose=False)
    surf = vmec.boundary
    objective_tuple = [(vmec.aspect, 4, 1)]
    if QA_or_QH: objective_tuple.append((vmec.mean_iota, 0.4, 1))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))

    # Finite Beta Virtual Casing Principle
    if finite_beta:
        if mpi.proc0_world: print('Running the virtual casing calculation')
        vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
        total_current = vmec.external_current() / (2 * surf.nfp)
        initial_current = total_current / ncoils * 1e-5
    else:
        initial_current = 1

    # Stage 2
    base_curves = create_equally_spaced_curves(ncoils, vmec.indata.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    if finite_beta:
        base_currents = [Current(initial_current) * 1e5 for i in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
    else:
        base_currents = [Current(initial_current) * 1e5 for i in range(ncoils)]
        base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, vmec.indata.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if finite_beta: Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    else: Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(base_curves))
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]
    JF = Jf \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + LENGTH_CON_WEIGHT * QuadraticPenalty(sum(Jls[i] for i in range(len(base_curves))), LENGTHBOUND) \
    + ARCLENGTH_WEIGHT * sum(Jals)

    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2

    def set_dofs(x0):
        if np.sum(JF.x!=x0[:-number_vmec_dofs])>0:
            JF.x = x0[:-number_vmec_dofs]
        if np.sum(prob.x!=x0[-number_vmec_dofs:])>0:
            prob.x = x0[-number_vmec_dofs:]
            if finite_beta:
                vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
                Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
                JF.opts[0].opts[0].opts[0] = Jf
        bs.set_points(surf.gamma().reshape((-1, 3)))

    def fun_J(dofs_vmec, dofs_coils):
        print(f'    processor {MPI.COMM_WORLD.Get_rank()} running fun_J')
        set_dofs(np.concatenate((np.ravel(dofs_coils), np.ravel(dofs_vmec))))
        if only_grad_stage2:
            J_stage_1 = 0
        else:
            J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        return J

    def grad_fun_analytical(x0, finite_difference_rel_step=rel_step, finite_difference_abs_step=abs_step, derivative_algorithm=derivative_algorithm):
        set_dofs(x0)
        dofs_vmec = prob.x
        dofs_coils = JF.x
        ## Finite differences for the second-stage objective function
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        if finite_beta:
            opt = make_optimizable(fun_J, dofs_vmec, dofs_coils, dof_indicators=["dof","non-dof"])
            grad_with_respect_to_surface = np.empty(len(dofs_vmec))
            with MPIFiniteDifference(opt.J, mpi, diff_method=derivative_algorithm, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
                if mpi.proc0_world:
                    grad_with_respect_to_surface = prob_jacobian.jac(dofs_vmec, dofs_coils)[0]
            mpi.comm_world.Bcast(grad_with_respect_to_surface, root=0)
            grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
        else:
            grad = np.array([0]*len(x0))
            with MPIFiniteDifference(prob.objective, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=derivative_algorithm) as prob_jacobian:
                if mpi.proc0_world:
                    if only_grad_stage2:
                        prob_dJ = 0
                    else:
                        prob_dJ = prob_jacobian.jac(prob.x)
                    surface = surf
                    bs.set_points(surface.gamma().reshape((-1, 3)))
                    ## Mixed term - derivative of squared flux with respect to the surface shape
                    n = surface.normal()
                    absn = np.linalg.norm(n, axis=2)
                    B = bs.B().reshape((nphi, ntheta, 3))
                    dB_by_dX = bs.dB_by_dX().reshape((nphi, ntheta, 3, 3))
                    Bcoil = bs.B().reshape(n.shape)
                    unitn = n * (1./absn)[:, :, None]
                    Bcoil_n = np.sum(Bcoil*unitn, axis=2)
                    mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
                    B_n = Bcoil_n
                    B_diff = Bcoil
                    B_N = np.sum(Bcoil * n, axis=2)
                    # assert Jf.local
                    dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
                    dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
                    deriv = surface.dnormal_by_dcoeff_vjp(dJdN/(nphi*ntheta)) + surface.dgamma_by_dcoeff_vjp(dJdx/(nphi*ntheta))
                    mixed_dJ = Derivative({surface: deriv})(surface)
                    ## Put both gradients together
                    grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
                    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
                mpi.comm_world.Bcast(grad, root=0)
        return grad

    def f(x0, gradient=True):
        set_dofs(x0)
        if only_grad_stage2:
            J_stage_1 = 0
        else:
            J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        if gradient: grad = grad_fun_analytical(x0)
        else: grad = 0
        return J, grad

    if mpi.proc0_world: print(f"""    ########################################################################
    ######## {QA_or_QH} ############################################################
    ########################################################################""")
    dofs = np.concatenate((JF.x, vmec.x))
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    if mpi.proc0_world: print("    Calculating f")
    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    if mpi.proc0_world: print("    Calculating epsilon")
    err_array = []
    for eps in eps_array:
        start_time = time.time()
        J1, _ = f(dofs + eps*h, gradient=False)
        J2, _ = f(dofs - eps*h, gradient=False)
        err = np.abs((J1-J2)/(2*eps) - dJh)
        end_time = time.time()
        if mpi.proc0_world:print(f"        eps={eps:.2}, err={err:.3}, time={(end_time-start_time):.3}s")
        err_array.append(err)
    if mpi.proc0_world:
        # Plot and save results
        if only_grad_stage2: plt.loglog(eps_array, err_array, 'o-', label=QA_or_QH, linewidth=2.0)
        else:                plt.loglog(eps_array, err_array, 'o-', label=QA_or_QH, linewidth=2.0)

print('hellow')
# plt.gca().invert_xaxis()
plt.xlabel('step size $\Delta x$', fontsize=22)
if only_grad_stage2:
    plt.ylabel("$|\Delta J_2/\Delta x - J_2'(x)|$", fontsize=22)
else:
    plt.ylabel("$|\Delta J/\Delta x - J'(x)|$", fontsize=22)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
# ax.legend(fontsize=20)
plt.legend(fontsize=20)
if only_grad_stage2:
    fname = f"taylor_test_result_old_J2.png"
else:
    fname = f"taylor_test_result_old.png"
plt.savefig(fname, dpi=250)
plt.close()
# plt.show()

# Remove spurious files
if mpi.proc0_world:
    for objective_file in glob.glob(f"jac_log*"): os.remove(objective_file)
    for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
    for objective_file in glob.glob(f"threed*"): os.remove(objective_file)
    for objective_file in glob.glob(f"mercier*"): os.remove(objective_file)
    for objective_file in glob.glob(f"fort*"): os.remove(objective_file)
    for objective_file in glob.glob(f"wout*"): os.remove(objective_file)
    for objective_file in glob.glob(f"input*"): os.remove(objective_file)
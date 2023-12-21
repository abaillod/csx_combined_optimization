#!/usr/bin/env python
import os
import time
import glob
from pathlib import Path
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

# Define absolute steps
abs_step_array = [1e-2,1e-3,1e-4,1e-5,1e-6]
rel_step_value = 1e-5

# Input parameters
QA_or_QHs = ['QH','QA']
derivative_algorithms = ["forward","centered"]
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

this_path = Path(__file__).parent.resolve()
OUT_DIR = os.path.join(this_path,f"output")
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

for QA_or_QH in QA_or_QHs:
    if finite_beta: vmec_file = os.path.join(this_path,f'input.precise{QA_or_QH}_FiniteBeta')
    else: vmec_file = os.path.join(this_path,f'input.precise{QA_or_QH}')
    vmec = Vmec(vmec_file, nphi=nphi, ntheta=ntheta, mpi=mpi, verbose=False, range_surface='half period')
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

    def fun(x0):
        set_dofs(x0)
        J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        return J
    
    def fun_new(prob, coils_prob):
        J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        return J

    def fun_J(dofs_vmec, dofs_coils):
        print(f'    processor {MPI.COMM_WORLD.Get_rank()} running fun_J')
        set_dofs(np.concatenate((np.ravel(dofs_coils), np.ravel(dofs_vmec))))
        J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        return J

    def grad_fun_analytical(x0, finite_difference_rel_step=0, finite_difference_abs_step=1e-7, derivative_algorithm="forward"):
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
            # prob_jacobian = FiniteDifference(opt.J, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=derivative_algorithm)
                    grad_with_respect_to_surface = prob_jacobian.jac(dofs_vmec, dofs_coils)[0]
            mpi.comm_world.Bcast(grad_with_respect_to_surface, root=0)
            # pprint(f'grad_with_respect_to_surface={grad_with_respect_to_surface}')
            grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
            # alternative_grad = prob_jacobian.jac(dofss)[0]
        else:
            grad = np.array([0]*len(x0))
            with MPIFiniteDifference(prob.objective, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=derivative_algorithm) as prob_jacobian:
            # prob_jacobian = FiniteDifference(prob.objective, x0=dofs_vmec, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=derivative_algorithm)
            # prob_dJ = prob_jacobian.jac(dofs_vmec)
                if mpi.proc0_world:
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

    def grad_fun_numerical(x0, diff_method="forward", abs_step = 1e-7, rel_step = 0):
        set_dofs(x0)
        grad = np.zeros(len(x0),)
        steps = finite_difference_steps(x0, abs_step=abs_step, rel_step=rel_step)
        if diff_method == "centered":
            for j in range(len(x0)):
                # print(f'FiniteDifference iteration {j}/{len(x0)}')
                x = np.copy(x0)
                x[j] = x0[j] + steps[j]
                fplus = fun(x)
                x[j] = x0[j] - steps[j]
                fminus = fun(x)
                grad[j] = (fplus - fminus) / (2 * steps[j])
        elif diff_method == "forward":
            f0 = fun(x0)
            for j in range(len(x0)):
                # print(f'FiniteDifference iteration {j}/{len(x0)}')
                x = np.copy(x0)
                x[j] = x0[j] + steps[j]
                fplus = fun(x)
                grad[j] = (fplus - f0) / steps[j]
        return grad

    # Set degrees of freedom
    dofs = np.concatenate((JF.x, vmec.x))

    # Perform regression test
    start_outer = time.time()
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.subplot(111)
    plt.title(QA_or_QH, fontsize=24)
    for derivative_algorithm in derivative_algorithms:
        sqrt_squared_diff_grad_with_respect_to_coils_array=[]
        sqrt_squared_diff_grad_with_respect_to_surface_array=[]
        for abs_step in abs_step_array:
            set_dofs(dofs)
            if mpi.proc0_world: print(f'{QA_or_QH} with abs_step={abs_step}')

            # gradNumerical = np.empty(len(dofs))
            # # opt = make_optimizable(fun, dofs, dof_indicators=["dof"])
            # opt = make_optimizable(fun_new, prob, JF)
            # with MPIFiniteDifference(opt.J, mpi, diff_method=derivative_algorithm, abs_step=abs_step, rel_step=rel_step_value) as fd:
            #     if mpi.proc0_world:
            #         start_inner_inner = time.time()
            #         # if mpi.proc0_world: print(f'  gradNumerical')
            #         gradNumerical = np.array(fd.jac()[0])
            #         print(f'   gradNumerical took {(time.time()-start_inner_inner):.2}s')
            # mpi.comm_world.Bcast(gradNumerical, root=0)

            gradNumerical = grad_fun_numerical(x0=dofs, diff_method=derivative_algorithm, abs_step=abs_step, rel_step=rel_step_value)

            gradNumerical_with_respect_to_coils = gradNumerical[:-number_vmec_dofs]
            gradNumerical_with_respect_to_surface = gradNumerical[-number_vmec_dofs:]

            ############

            start_inner_inner = time.time()
            # if mpi.proc0_world: print(f'  gradAnalytical')
            gradAnalytical = grad_fun_analytical(dofs, finite_difference_rel_step=rel_step_value, finite_difference_abs_step=abs_step, derivative_algorithm=derivative_algorithm)
            if mpi.proc0_world: print(f'   gradAnalytical took {(time.time()-start_inner_inner):.2}s')
            gradAnalytical_with_respect_to_coils = gradAnalytical[:-number_vmec_dofs]
            gradAnalytical_with_respect_to_surface = gradAnalytical[-number_vmec_dofs:]

            ############

            sqrt_squared_diff_grad_with_respect_to_coils = np.sqrt(np.sum((gradAnalytical_with_respect_to_coils - gradNumerical_with_respect_to_coils)**2))
            sqrt_squared_diff_grad_with_respect_to_surface = np.sqrt(np.sum((gradAnalytical_with_respect_to_surface - gradNumerical_with_respect_to_surface)**2))

            sqrt_squared_diff_grad_with_respect_to_coils_array.append(sqrt_squared_diff_grad_with_respect_to_coils)
            sqrt_squared_diff_grad_with_respect_to_surface_array.append(sqrt_squared_diff_grad_with_respect_to_surface)

            # if mpi.proc0_world:
            #     print(f' Inner abs_step took {time.time()-start_inner}s')

        if mpi.proc0_world:
            # print(f'Outer abs_step loop took {time.time()-start_outer}s')
            # Print results
            print(f'abs_step_array={abs_step_array}')
            print(f'RMS differences grad coils with diff_method={derivative_algorithm}={sqrt_squared_diff_grad_with_respect_to_coils_array}')
            print(f'RMS differences grad surface with diff_method={derivative_algorithm}={sqrt_squared_diff_grad_with_respect_to_surface_array}')
            # Plot and save results
            plt.loglog(abs_step_array, sqrt_squared_diff_grad_with_respect_to_coils_array, 'o-', label=r'$\Delta J(x_{\mathrm{coils}})$'+f' ({derivative_algorithm})', linewidth=2.0)
            plt.loglog(abs_step_array, sqrt_squared_diff_grad_with_respect_to_surface_array, 'o-', label=r'$\Delta J(x_{\mathrm{surface}})$'+f' ({derivative_algorithm})', linewidth=2.0)
                
    # plt.gca().invert_xaxis()
    plt.xlabel('step size $\Delta x_i$', fontsize=22)
    plt.ylabel('RMS($\Delta J$)', fontsize=22)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    ax.legend(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"single_stage_test_result_{QA_or_QH}.png", dpi=250)
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
"""
Optimization of CSSC rescaled

"""

from simsopt.objectives import Weight
import numpy as np

# User inputs
# ===========
inputs = dict()
inputs['coils_objective_weight'] = Weight(1e2)
inputs['CS_THRESHOLD'] = 0.08
inputs['CS_WEIGHT'] = Weight(1)
inputs['CC_THRESHOLD'] = 0.08
inputs['CC_WEIGHT'] = Weight(1)
inputs['directory'] = 'runs/single_stage_constrained_optimization_002'

# SURFACE
inputs['vmec'] = dict()
inputs['vmec']['filename'] = 'inputs/vmec_inputs/input.vacuum_cssc_scaled' 
inputs['vmec']['verbose'] = False
inputs['vmec']['nphi'] = 34
inputs['vmec']['ntheta'] = 34
inputs['vmec']['internal_mpol'] = 5
inputs['vmec']['internal_ntor'] = 5

inputs['vmec']['dofs'] = dict()
inputs['vmec']['dofs']['mpol'] = 2
inputs['vmec']['dofs']['ntor'] = 2

inputs['vmec']['target'] = dict()
inputs['vmec']['target']['aspect_ratio'] = 2
inputs['vmec']['target']['aspect_ratio_weight'] = Weight(1e3)
inputs['vmec']['target']['aspect_ratio_constraint_type'] = 'max'               # Identity for target, max or min for constraint
inputs['vmec']['target']['iota'] = -0.18
inputs['vmec']['target']['iota_weight'] = Weight(1e3)
inputs['vmec']['target']['iota_constraint_type'] = 'max'               # Identity for target, max or min for constraint
inputs['vmec']['target']['qa_surface'] = np.array([0.25, 0.5, 0.75, 1]) # Weight for QA is 1.
inputs['vmec']['target']['qa_ntheta'] = 63
inputs['vmec']['target']['qa_nphi'] = 64

# COILS
## Interlinked (IL) and Poloidal field (PF) coils related inputs
inputs['cnt_coils'] = dict()
inputs['cnt_coils']['geometry'] = dict()
inputs['cnt_coils']['geometry']['filename'] = 'flux_100_bs_cssc_cssc.json'

inputs['cnt_coils']['dofs'] = dict()
inputs['cnt_coils']['dofs']['IL_order'] = 2 # In G. Rawlinson input, this was 7
inputs['cnt_coils']['dofs']['IL_geometry_free'] = True
inputs['cnt_coils']['dofs']['PF_current_free'] = True

inputs['cnt_coils']['target'] = dict()
inputs['cnt_coils']['target']['IL_length'] = 3
inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1)
inputs['cnt_coils']['target']['IL_length_constraint_type'] = 'max'
inputs['cnt_coils']['target']['IL_msc_threshold'] = 10
inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(0)
inputs['cnt_coils']['target']['IL_maxc_threshold'] = 20
inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(1)
inputs['cnt_coils']['target']['PF_current_threshold'] = 1E6 
inputs['cnt_coils']['target']['PF_current_weight'] = Weight(1)

# NUMERICS
inputs['numerics'] = dict()
inputs['numerics']['MAXITER_stage_1'] = 100 # NUmber of iteration for initial stage two optimization
inputs['numerics']['MAXITER_stage_2'] = 250 # NUmber of iteration for combined optimization
inputs['numerics']['fndiff_method'] = "forward"
inputs['numerics']['finite_difference_abs_step'] = 1E-8
inputs['numerics']['finite_difference_rel_step'] = 1E-5
inputs['numerics']['JACOBIAN_THRESHOLD'] = 1e9 # In G Rawlinson input, this was set to 1E2
inputs['numerics']['algorithm'] = 'BFGS'


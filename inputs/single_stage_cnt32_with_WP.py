"""
Optimization of CNT 32

This input optimizes the CNT 32 configuration for better
quasisymmetry, while constraining the rotational transform
and the aspect ratio to remain above a threshold
"""

from simsopt.objectives import Weight
import numpy as np

# User inputs
# ===========
inputs = dict()
inputs['coils_objective_weight'] = Weight(1E3)
inputs['CS_THRESHOLD'] = 0.06
inputs['CS_WEIGHT'] = Weight(1)
inputs['CC_THRESHOLD'] = 0.06
inputs['CC_WEIGHT'] = Weight(1)
inputs['directory'] = 'runs/opt_02/cnt32'

# SURFACE
inputs['vmec'] = dict()
inputs['vmec']['filename'] = 'inputs/vmec_inputs/input.cnt32'
inputs['vmec']['verbose'] = False
inputs['vmec']['nphi'] = 34
inputs['vmec']['ntheta'] = 34
inputs['vmec']['internal_mpol'] = 5
inputs['vmec']['internal_ntor'] = 5


inputs['vmec']['dofs'] = dict()
inputs['vmec']['dofs']['mpol'] = 1
inputs['vmec']['dofs']['ntor'] = 1

inputs['vmec']['target'] = dict()
inputs['vmec']['target']['aspect_ratio'] = 2
inputs['vmec']['target']['aspect_ratio_weight'] = Weight(5)
inputs['vmec']['target']['aspect_ratio_constraint_type'] = 'max'               # Identity for target, max or min for constraint
inputs['vmec']['target']['iota'] = -0.18
inputs['vmec']['target']['iota_weight'] = Weight(5)
inputs['vmec']['target']['iota_constraint_type'] = 'max'               # Identity for target, max or min for constraint
inputs['vmec']['target']['qa_surface'] = np.linspace(0,1,10,endpoint=True) # Weight for QA is 1.
inputs['vmec']['target']['qa_ntheta'] = 63
inputs['vmec']['target']['qa_nphi'] = 64
inputs['vmec']['target']['volume'] = 0.15
inputs['vmec']['target']['volume_weight'] = Weight(0)
inputs['vmec']['target']['volume_constraint_type'] = 'min'

# COILS
## Interlinked (IL) and Poloidal field (PF) coils related inputs
inputs['cnt_coils'] = dict()
inputs['cnt_coils']['geometry'] = dict()
inputs['cnt_coils']['geometry']['filename'] = 'inputs/coil_inputs/biotsavart_cnt32.json'

inputs['cnt_coils']['dofs'] = dict()
inputs['cnt_coils']['dofs']['IL_order'] = 7 # In G. Rawlinson input, this was 7
inputs['cnt_coils']['dofs']['IL_geometry_free'] = True
inputs['cnt_coils']['dofs']['PF_current_free'] = True
inputs['cnt_coils']['dofs']['R00_free'] = False

inputs['cnt_coils']['target'] = dict()
inputs['cnt_coils']['target']['IL_length'] = 4.0
inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1E-2)
inputs['cnt_coils']['target']['IL_length_constraint_type'] = 'max'
inputs['cnt_coils']['target']['IL_msc_threshold'] = 60
inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(0.01)
inputs['cnt_coils']['target']['IL_maxc_threshold'] = 75
inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(0.01)
inputs['cnt_coils']['target']['PF_current_threshold'] = 0 
inputs['cnt_coils']['target']['PF_current_weight'] = Weight(0)
inputs['cnt_coils']['target']['IL_vessel_threshold'] = 0.16
inputs['cnt_coils']['target']['IL_vessel_weight'] = Weight(0)
inputs['cnt_coils']['target']['IL_maxR_threshold'] = 0.65
inputs['cnt_coils']['target']['IL_maxR_weight'] = Weight(10)
inputs['cnt_coils']['target']['IL_maxZ_threshold'] = 0.75
inputs['cnt_coils']['target']['IL_maxZ_weight'] = Weight(10)
inputs['cnt_coils']['target']['arclength_weight'] = Weight(1E-4)


## Windowpane coils related inputs
inputs['wp_coils'] = dict()
inputs['wp_coils']['geometry'] = dict()
inputs['wp_coils']['geometry']['filename'] = 'inputs/coil_inputs/windowpanes/wp_set_03.json'  # if None, coils are initialized 
                                                  # according to inputs below
inputs['wp_coils']['geometry']['n_base_coils'] = 1

inputs['wp_coils']['target'] = dict()
inputs['wp_coils']['target']['length'] = 10.0
inputs['wp_coils']['target']['length_weight'] = Weight(1E-2)
inputs['wp_coils']['target']['length_constraint_type'] = 'max'
inputs['wp_coils']['target']['msc_threshold'] = 60
inputs['wp_coils']['target']['msc_weight'] = Weight(0)
inputs['wp_coils']['target']['maxc_threshold'] = 75
inputs['wp_coils']['target']['maxc_weight'] = Weight(0)
inputs['wp_coils']['target']['current_threshold'] = 1E+5 
inputs['wp_coils']['target']['current_weight'] = Weight(0)
inputs['wp_coils']['target']['winding_surface_weight'] = Weight(0)

inputs['wp_coils']['dofs'] = dict()
inputs['wp_coils']['dofs']['name'] = ['z0', 'xc(1)', 'zs(1)']


# NUMERICS
inputs['numerics'] = dict()
inputs['numerics']['MAXITER_stage_2'] = 0 # NUmber of iteration for initial stage two optimization
inputs['numerics']['MAXITER_single_stage'] = 5000 # NUmber of iteration for combined optimization
inputs['numerics']['fndiff_method'] = "forward"
inputs['numerics']['finite_difference_abs_step'] = 1E-7
inputs['numerics']['finite_difference_rel_step'] = 0
inputs['numerics']['JACOBIAN_THRESHOLD'] = 1e9 # In G Rawlinson input, this was set to 1E2
inputs['numerics']['algorithm'] = 'BFGS'


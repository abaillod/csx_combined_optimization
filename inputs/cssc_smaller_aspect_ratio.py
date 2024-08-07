from simsopt.objectives import Weight
import numpy as np

# User inputs
# ===========
inputs = dict()
inputs['coils_objective_weight'] = Weight(1e3)
inputs['CS_THRESHOLD'] = 0.06
inputs['CS_WEIGHT'] = Weight(1e-1)
inputs['CC_THRESHOLD'] = 0.06
inputs['CC_WEIGHT'] = Weight(1e-1)
inputs['directory'] = 'runs/cssc_tight_aspect_ratio_002'

# SURFACE
inputs['vmec'] = dict()
inputs['vmec']['filename'] = 'runs/cssc_tight_aspect_ratio/input.final'
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
inputs['vmec']['target']['aspect_ratio_weight'] = Weight(10)
inputs['vmec']['target']['aspect_ratio_constraint_type'] = 'identity'               # Identity for target, max or min for constraint
inputs['vmec']['target']['iota'] = -0.18
inputs['vmec']['target']['iota_weight'] = Weight(10)
inputs['vmec']['target']['iota_constraint_type'] = 'identity'               # Identity for target, max or min for constraint
inputs['vmec']['target']['qa_surface'] = np.array([0.25, 0.5, 0.75, 1]) # Weight for QA is 1.
inputs['vmec']['target']['qa_ntheta'] = 63
inputs['vmec']['target']['qa_nphi'] = 64

# COILS
## Interlinked (IL) and Poloidal field (PF) coils related inputs
inputs['cnt_coils'] = dict()
inputs['cnt_coils']['geometry'] = dict()
inputs['cnt_coils']['geometry']['filename'] = 'inputs/flux_100_bs_cssc_cssc.json'

inputs['cnt_coils']['dofs'] = dict()
inputs['cnt_coils']['dofs']['IL_order'] = 2 # In G. Rawlinson input, this was 7
inputs['cnt_coils']['dofs']['IL_geometry_free'] = True
inputs['cnt_coils']['dofs']['PF_current_free'] = True

inputs['cnt_coils']['target'] = dict()
inputs['cnt_coils']['target']['IL_length'] = 3
inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1e-2)
inputs['cnt_coils']['target']['IL_length_constraint_type'] = 'max'
inputs['cnt_coils']['target']['IL_msc_threshold'] = 38.949180592254365
inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(1e-2)
inputs['cnt_coils']['target']['IL_maxc_threshold'] = 38.19183493544338
inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(1e-2)
inputs['cnt_coils']['target']['PF_current_threshold'] = 1E6 
inputs['cnt_coils']['target']['PF_current_weight'] = Weight(1e-2)

## Windowpane coils related inputs
inputs['wp_coils'] = dict()
inputs['wp_coils']['geometry'] = dict()
inputs['wp_coils']['geometry']['filename'] = None # if None, coils are initialized 
                                                  # according to inputs below
inputs['wp_coils']['geometry']['ncoil_per_row'] = 0 # total number of wp coils will be 
                                                    # nfp*size(Z0)*ncoil_per_row
                                                    # if zero, all WP penalty are unused
inputs['wp_coils']['geometry']['R0'] = 0.3
inputs['wp_coils']['geometry']['R1'] = 0.05
inputs['wp_coils']['geometry']['Z0'] = [0]

inputs['wp_coils']['dofs'] = dict()
inputs['wp_coils']['dofs']['order'] = 2
inputs['wp_coils']['dofs']['planar'] = True # Enforce coils to remain planar if True

inputs['wp_coils']['target'] = dict()
inputs['wp_coils']['target']['length'] = inputs['cnt_coils']['target']['IL_length'] / 3
inputs['wp_coils']['target']['length_weight'] = Weight(0)
inputs['wp_coils']['target']['length_constraint_type'] = 'max'
inputs['wp_coils']['target']['msc_threshold'] = 20
inputs['wp_coils']['target']['msc_weight'] = Weight(0)
inputs['wp_coils']['target']['maxc_threshold'] = 50
inputs['wp_coils']['target']['maxc_weight'] = Weight(0)
inputs['wp_coils']['target']['current_threshold'] = 1E+5 
inputs['wp_coils']['target']['current_weight'] = Weight(0)


# NUMERICS
inputs['numerics'] = dict()
inputs['numerics']['MAXITER_stage_1'] = 100 # NUmber of iteration for initial stage two optimization
inputs['numerics']['MAXITER_stage_2'] = 250 # NUmber of iteration for combined optimization
inputs['numerics']['fndiff_method'] = "forward"
inputs['numerics']['finite_difference_abs_step'] = 1E-8
inputs['numerics']['finite_difference_rel_step'] = 1E-5
inputs['numerics']['JACOBIAN_THRESHOLD'] = 1e3 # In G Rawlinson input, this was set to 1E2
inputs['numerics']['algorithm'] = 'BFGS'


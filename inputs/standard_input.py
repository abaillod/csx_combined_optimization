from simsopt.objectives import Weight
import numpy as np

# User inputs
# ===========
inputs = dict()
inputs['coils_objective_weight'] = Weight(1) # All coil-related penalties are scaled by this weight
inputs['CS_THRESHOLD'] = 0.06                # Minimum coil to surface distance
inputs['CS_WEIGHT'] = Weight(1E-1)           # Weight on coil-surface penalty
inputs['CC_THRESHOLD'] = 0.1                 # Minimum coil to coil distance
inputs['CC_WEIGHT'] = Weight(1E-1)           # Weight on coil-coil distance
inputs['directory'] = 'runs/test_dir'        # Name of output directory

# SURFACE
inputs['vmec'] = dict()
inputs['vmec']['filename'] = 'inputs/vmec_inputs/input.CSSCscaled3' # Input filename for VMEC
inputs['vmec']['verbose'] = False                # Set to True for additional VMEC messages (useful for debug)
inputs['vmec']['nphi'] = 34                      # VMEC toroidal resolution in real space
inputs['vmec']['ntheta'] = 34                    # VMEC poloidal resolution in real space
inputs['vmec']['internal_mpol'] = 2 
inputs['vmec']['internal_ntor'] = 2

inputs['vmec']['dofs'] = dict() 
inputs['vmec']['dofs']['mpol'] = 2 # VMEC boundary dofs with  m<=mpol will be unfixed
inputs['vmec']['dofs']['ntor'] = 2 # VMEC boundary dofs with |n|<= ntor will be unfixed

inputs['vmec']['target'] = dict()
inputs['vmec']['target']['aspect_ratio'] = 2                             # Target value for boundary aspect ratio
inputs['vmec']['target']['aspect_ratio_weight'] = Weight(1E+1)             # Weight for aspect ratio target
inputs['vmec']['target']['aspect_ratio_constraint_type'] = 'identity'               # Identity for target, max or min for constraint
inputs['vmec']['target']['iota'] = -0.2                                    # Target value for mean iota
inputs['vmec']['target']['iota_weight'] = Weight(1E+2)                     # Weight for iota target
inputs['vmec']['target']['iota_constraint_type'] = 'identity'               # Identity for target, max or min for constraint
inputs['vmec']['target']['qa_surface'] = np.linspace(0,1,11,endpoint=True) # Weight for QA is 1.
inputs['vmec']['target']['qa_ntheta'] = 63                                 # Poloidal resolution for QS surfaces
inputs['vmec']['target']['qa_nphi'] = 64                                   # Toroidal resolution for QS surfaces


# COILS
## Interlinked (IL) and Poloidal field (PF) coils related inputs
inputs['cnt_coils'] = dict()
inputs['cnt_coils']['geometry'] = dict()
inputs['cnt_coils']['geometry']['filename'] = 'inputs/coil_inputs/flux_100_bs_cssc_cssc.json' # Input file for IL and PF coils initial guess

inputs['cnt_coils']['dofs'] = dict()
inputs['cnt_coils']['dofs']['IL_order'] = 2            # The xn, yn, zn, with n<=IL_order are unfixed 
inputs['cnt_coils']['dofs']['IL_geometry_free'] = True # Set to True to unfix IL coils geometry
inputs['cnt_coils']['dofs']['PF_current_free'] = True  # Set to True to unfix PF current

inputs['cnt_coils']['target'] = dict()
inputs['cnt_coils']['target']['IL_length'] = 3                   # Maximum length for IL coils
inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1E-5) # Weight on IL length penalty
inputs['cnt_coils']['target']['IL_length_constraint_type'] = 'max' # Can be 'max', 'min', or 'identity'
inputs['cnt_coils']['target']['IL_msc_threshold'] = 10           # Maximum mean curvature of IL coils
inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(1)       # Weight on IL mean curvature penalty
inputs['cnt_coils']['target']['IL_maxc_threshold'] = 20          # Maximum local curvature of IL coils
inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(1)      # Weight on IL local curvature penalty
inputs['cnt_coils']['target']['PF_current_threshold'] = 1E+9     # Maximum PF current value
inputs['cnt_coils']['target']['PF_current_weight'] = Weight(1)   # Weight on PF current penalty
inputs['cnt_coils']['target']['IL_maxR_threshold'] = 0.65     # Max radial position of coils - can be used to constrain coils to remain in the vessel
inputs['cnt_coils']['target']['IL_maxR_weight'] = Weight(1)   # Weight on max radial position
inputs['cnt_coils']['target']['IL_maxZ_threshold'] = 0.75     # Max vertical position of coils
inputs['cnt_coils']['target']['IL_maxZ_weight'] = Weight(1)   # Weight on max vertical position

# NUMERICS
inputs['numerics'] = dict()
inputs['numerics']['MAXITER_stage_2'] = 10 # NUmber of iteration for initial stage two optimization
inputs['numerics']['MAXITER_single_stage'] = 10 # NUmber of iteration for combined optimization
inputs['numerics']['fndiff_method'] = "forward" # Method to evaluate the finite differences. Either 'forward', 'centered', or 'backward'
inputs['numerics']['finite_difference_abs_step'] = 1E-7     # Default value is 1E-7
inputs['numerics']['finite_difference_rel_step'] = 0        # Default value is 0
inputs['numerics']['JACOBIAN_THRESHOLD'] = 100
inputs['numerics']['taylor_test'] = False

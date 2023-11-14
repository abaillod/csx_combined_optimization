from simsopt.objectives import Weight
import numpy as np

# User inputs
# ===========
inputs = dict()
inputs['coils_objective_weight'] = Weight(1) # All coil-related penalties are scaled by this weight
inputs['CS_THRESHOLD'] = 0.06                # Minimum coil to surface distance
inputs['CS_WEIGHT'] = Weight(1)           # Weight on coil-surface penalty
inputs['CC_THRESHOLD'] = 0.1                 # Minimum coil to coil distance
inputs['CC_WEIGHT'] = Weight(1)           # Weight on coil-coil distance
inputs['directory'] = 'runs/stage_2_order_2'        # Name of output directory

# SURFACE
inputs['vmec'] = dict()
inputs['vmec']['filename'] = 'runs/stage_1_mpol=ntor=3/input.final' # Input filename for VMEC
inputs['vmec']['verbose'] = False                # Set to True for additional VMEC messages (useful for debug)
inputs['vmec']['nphi'] = 34                      # VMEC toroidal resolution in real space
inputs['vmec']['ntheta'] = 34                    # VMEC poloidal resolution in real space

inputs['vmec']['dofs'] = dict() 
inputs['vmec']['dofs']['mpol'] = -1 # VMEC boundary dofs with  m<=mpol will be unfixed
inputs['vmec']['dofs']['ntor'] = -1 # VMEC boundary dofs with |n|<= ntor will be unfixed

inputs['vmec']['target'] = dict()
inputs['vmec']['target']['aspect_ratio'] = 3.5                             # Target value for boundary aspect ratio
inputs['vmec']['target']['aspect_ratio_weight'] = Weight(0)             # Weight for aspect ratio target
inputs['vmec']['target']['iota'] = -0.2                                    # Target value for mean iota
inputs['vmec']['target']['iota_weight'] = Weight(0)                     # Weight for iota target
inputs['vmec']['target']['qa_surface'] = np.linspace(0,1,11,endpoint=True) # Weight for QA is 1.
inputs['vmec']['target']['qa_ntheta'] = 63                                 # Poloidal resolution for QS surfaces
inputs['vmec']['target']['qa_nphi'] = 64                                   # Toroidal resolution for QS surfaces


# COILS
## Interlinked (IL) and Poloidal field (PF) coils related inputs
inputs['cnt_coils'] = dict()
inputs['cnt_coils']['geometry'] = dict()
inputs['cnt_coils']['geometry']['filename'] = 'runs/stage_2_order_1/coils/bs_post_stage_2.json' # Input file for IL and PF coils initial guess

inputs['cnt_coils']['dofs'] = dict()
inputs['cnt_coils']['dofs']['IL_order'] = 2            # The xn, yn, zn, with n<=IL_order are unfixed 
inputs['cnt_coils']['dofs']['IL_geometry_free'] = True # Set to True to unfix IL coils geometry
inputs['cnt_coils']['dofs']['PF_current_free'] = True  # Set to True to unfix PF current

inputs['cnt_coils']['target'] = dict()
inputs['cnt_coils']['target']['IL_length'] = 3                   # Maximum length for IL coils
inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1E-1) # Weight on IL length penalty
inputs['cnt_coils']['target']['IL_msc_threshold'] = 50           # Maximum mean curvature of IL coils
inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(1e3)       # Weight on IL mean curvature penalty
inputs['cnt_coils']['target']['IL_maxc_threshold'] = 50          # Maximum local curvature of IL coils
inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(1e3)      # Weight on IL local curvature penalty
inputs['cnt_coils']['target']['PF_current_threshold'] = 1E+9     # Maximum PF current value
inputs['cnt_coils']['target']['PF_current_weight'] = Weight(1e3)   # Weight on PF current penalty

## Windowpane coils related inputs
inputs['wp_coils'] = dict()
inputs['wp_coils']['geometry'] = dict()
inputs['wp_coils']['geometry']['filename'] = None # if None, coils are initialized 
                                                  # according to inputs below
inputs['wp_coils']['geometry']['ncoil_per_row'] = 0 # total number of wp coils will be 
                                                    # nfp*size(Z0)*ncoil_per_row
inputs['wp_coils']['geometry']['R0'] = 0.3  # Initial radial position of WP coils
inputs['wp_coils']['geometry']['R1'] = 0.05 # Initial radius of WP coils
inputs['wp_coils']['geometry']['Z0'] = [0]  # Number of "rows" of WP coils

inputs['wp_coils']['dofs'] = dict()
inputs['wp_coils']['dofs']['order'] = 2     # The xn, yn, zn with |n|<=order will be unfixed
inputs['wp_coils']['dofs']['planar'] = True # Enforce coils to remain planar if True (fix all yn)

inputs['wp_coils']['target'] = dict()
inputs['wp_coils']['target']['length'] = inputs['cnt_coils']['target']['IL_length'] / 3 # max length for WP coils
inputs['wp_coils']['target']['length_weight'] = Weight(1E-1)  # Weight on WP length penalty
inputs['wp_coils']['target']['msc_threshold'] = 20            # Maximum mean curvature of WP coils
inputs['wp_coils']['target']['msc_weight'] = Weight(1e3)        # Weight on WP coils mean curvature penalty 
inputs['wp_coils']['target']['maxc_threshold'] = 50           # Maximum local curvature of WP coils
inputs['wp_coils']['target']['maxc_weight'] = Weight(1e3)       # Weight on WP local curvature
inputs['wp_coils']['target']['current_threshold'] = 1E+5      # Maximum current in WP coils
inputs['wp_coils']['target']['current_weight'] = Weight(1E-5) # Weight on WP maximum current penalty


# NUMERICS
inputs['numerics'] = dict()
inputs['numerics']['MAXITER_stage_1'] = 100 # NUmber of iteration for initial stage two optimization
inputs['numerics']['MAXITER_stage_2'] = 0 # NUmber of iteration for combined optimization
inputs['numerics']['fndiff_method'] = "forward"
inputs['numerics']['finite_difference_abs_step'] = 1E-8
inputs['numerics']['finite_difference_rel_step'] = 1E-5
inputs['numerics']['JACOBIAN_THRESHOLD'] = 100

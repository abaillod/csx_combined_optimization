from simsopt.objectives import Weight
import numpy as np

def set_default(inputs):
    if 'coils_objective_weight' not in inputs.keys():
        inputs['coils_objective_weight'] = Weight(1)
    if 'CS_THRESHOLD' not in inputs.keys():
        inputs['CS_THRESHOLD'] = 0
    if 'CS_WEIGHT' not in inputs.keys():
        inputs['CS_WEIGHT'] = Weight(0)
    if 'CC_THRESHOLD' not in inputs.keys():
        inputs['CC_THRESHOLD'] = 0
    if 'CC_WEIGHT' not in inputs.keys():
        inputs['CC_WEIGHT'] = Weight(0)
    if 'directory' not in inputs.keys():
        raise ValueError('No specified directory!')

    if 'vmec' not in inputs.keys():
        inputs['vmec'] = dict()
    if 'filename' not in inputs['vmec'].keys():
        inputs['vmec']['filename'] = 'input.vacuum_cssc_scaled'
    if 'verbose' not in inputs['vmec'].keys():
        inputs['vmec']['verbose'] = False
    if 'nphi' not in inputs['vmec'].keys():
        inputs['vmec']['nphi'] = 34  
    if 'ntheta' not in inputs['vmec'].keys():
        inputs['vmec']['ntheta'] = 34
    if 'internal_mpol' not in inputs['vmec'].keys():
        inputs['vmec']['internal_mpol'] = 8
    if 'internal_ntor' not in inputs['vmec'].keys():
        inputs['vmec']['internal_ntor'] = 8

 
    if 'dofs' not in inputs['vmec'].keys():
        inputs['vmec']['dofs'] = dict()
    if 'mpol' not in inputs['vmec']['dofs'].keys():
        inputs['vmec']['dofs']['mpol'] = 2
    if 'ntor' not in inputs['vmec']['dofs'].keys():
        inputs['vmec']['dofs']['ntor'] = 2

    if 'target' not in inputs['vmec'].keys():
        inputs['vmec']['target'] = dict()
    if 'aspect_ratio' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['aspect_ratio'] = 3.5 
    if 'aspect_ratio_weight' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['aspect_ratio_weight'] = Weight(1) 
    if 'aspect_ratio_constraint_type' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['aspect_ratio_constraint_type'] = 'identity'
    if 'iota' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['iota'] = -0.2 
    if 'iota_weight' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['iota_weight'] = Weight(1) 
    if 'iota_constraint_type' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['iota_constraint_type'] = 'identity'
    if 'qa_surface' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['qa_surface'] = np.array([1]) 
    if 'qa_ntheta' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['qa_ntheta'] = 63
    if 'qa_ntheta' not in inputs['vmec']['target'].keys():
        inputs['vmec']['target']['qa_ntheta'] = 64 



    # COILS
    if 'cnt_coils' not in inputs.keys():
        inputs['cnt_coils'] = dict()
    if 'geometry' not in inputs['cnt_coils'].keys():
        inputs['cnt_coils']['geometry'] = dict()
    if 'filename' not in inputs['cnt_coils']['geometry'].keys():
        inputs['cnt_coils']['geometry']['filename'] = 'flux_100_bs_cssc_cssc.json' # Input file for IL and PF coils initial guess

    if 'dofs' not in inputs['cnt_coils'].keys():
        inputs['cnt_coils']['dofs'] = dict()
    if 'IL_order' not in inputs['cnt_coils']['dofs'].keys():
        inputs['cnt_coils']['dofs']['IL_order'] = 2            # The xn, yn, zn, with n<=IL_order are unfixed 
    if 'IL_geometry_free' not in inputs['cnt_coils']['dofs'].keys():
        inputs['cnt_coils']['dofs']['IL_geometry_free'] = True # Set to True to unfix IL coils geometry
    if 'PF_current_free' not in inputs['cnt_coils']['dofs'].keys():
        inputs['cnt_coils']['dofs']['PF_current_free'] = True  # Set to True to unfix PF current
    
    if 'target' not in inputs['cnt_coils'].keys():
        inputs['cnt_coils']['target'] = dict()
    if 'IL_length' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_length'] = 3                   # Maximum length for IL coils
    if 'IL_length_weight' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_length_weight'] = Weight(1E-5) # Weight on IL length penalty
    if 'IL_msc_threshold' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_msc_threshold'] = 10           # Maximum mean curvature of IL coils
    if 'IL_msc_weight' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(1)       # Weight on IL mean curvature penalty
    if 'IL_maxc_threshold' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_maxc_threshold'] = 20          # Maximum local curvature of IL coils
    if 'IL_maxc_weight' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(1)      # Weight on IL local curvature penalty
    if 'PF_current_threshold' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['PF_current_threshold'] = 1E+9     # Maximum PF current value
    if 'PF_current_weight' not in inputs['cnt_coils']['target'].keys():
        inputs['cnt_coils']['target']['PF_current_weight'] = Weight(1)   # Weight on PF current penalty
        
    ## Windowpane coils related inputs
    if 'wp_coils' not in inputs.keys():
        inputs['wp_coils'] = dict()
    if 'geometry' not in inputs['wp_coils'].keys():
        inputs['wp_coils']['geometry'] = dict()
    if 'filename' not in inputs['wp_coils']['geometry'].keys():
        inputs['wp_coils']['geometry']['filename'] = None # if None, coils are initialized 
                                                          # according to inputs below
    if 'ncoil_per_row' not in inputs['wp_coils']['geometry'].keys():
        inputs['wp_coils']['geometry']['ncoil_per_row'] = 2 # total number of wp coils will be 
                                                            # nfp*size(Z0)*ncoil_per_row
    if 'R0' not in inputs['wp_coils']['geometry'].keys():
        inputs['wp_coils']['geometry']['R0'] = 0.3  # Initial radial position of WP coils
    if 'R1' not in inputs['wp_coils']['geometry'].keys():
        inputs['wp_coils']['geometry']['R1'] = 0.05 # Initial radius of WP coils
    if 'Z0' not in inputs['wp_coils']['geometry'].keys():
        inputs['wp_coils']['geometry']['Z0'] = [0]  # Number of "rows" of WP coils
    
    if 'dofs' not in inputs['wp_coils'].keys():    
        inputs['wp_coils']['dofs'] = dict()
    if 'order' not in inputs['wp_coils']['dofs'].keys():
        inputs['wp_coils']['dofs']['order'] = 2     # The xn, yn, zn with |n|<=order will be unfixed
    if 'planar' not in inputs['wp_coils']['dofs'].keys():
        inputs['wp_coils']['dofs']['planar'] = True # Enforce coils to remain planar if True (fix all yn)
        
    if 'target' not in inputs['wp_coils'].keys():
        inputs['wp_coils']['target'] = dict()
    if 'length' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['length'] = inputs['cnt_coils']['target']['IL_length'] / 3 # max length for WP coils
    if 'length_weight' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['length_weight'] = Weight(1E-5)  # Weight on WP length penalty
    if 'msc_threshold' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['msc_threshold'] = 20            # Maximum mean curvature of WP coils
    if 'msc_weight' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['msc_weight'] = Weight(0)        # Weight on WP coils mean curvature penalty 
    if 'maxc_threshold' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['maxc_threshold'] = 50           # Maximum local curvature of WP coils
    if 'maxc_weight' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['maxc_weight'] = Weight(0)       # Weight on WP local curvature
    if 'current_threshold' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['current_threshold'] = 1E+5      # Maximum current in WP coils
    if 'current_weight' not in inputs['wp_coils']['target'].keys():
        inputs['wp_coils']['target']['current_weight'] = Weight(1E-5) # Weight on WP maximum current penalty
        
        
    # NUMERICS
    if 'numerics' not in inputs.keys():
        inputs['numerics'] = dict()
    if 'MAXITER_stage_1' not in inputs['numerics'].keys():
        inputs['numerics']['MAXITER_stage_1'] = 10 # NUmber of iteration for initial stage two optimization
    if 'MAXITER_stage_2' not in inputs['numerics'].keys():
        inputs['numerics']['MAXITER_stage_2'] = 10 # NUmber of iteration for combined optimization
    if 'fndiff_method' not in inputs['numerics'].keys():
        inputs['numerics']['fndiff_method'] = "forward"
    if 'finite_difference_abs_step' not in inputs['numerics'].keys():
        inputs['numerics']['finite_difference_abs_step'] = 0
    if 'finite_difference_rel_step' not in inputs['numerics'].keys():
        inputs['numerics']['finite_difference_rel_step'] = 1E-5
    if 'JACOBIAN_THRESHOLD' not in inputs['numerics'].keys():
        inputs['numerics']['JACOBIAN_THRESHOLD'] = 100
    if 'algorithm' not in inputs['numerics'].keys():
        inputs['numerics']['algorithm'] = 'BFGS'

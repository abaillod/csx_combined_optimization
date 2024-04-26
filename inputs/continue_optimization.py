import pickle
from simsopt.objectives import Weight


fname = '../runs/opt_43/M=N=1/input.pckl'
with open(fname, 'rb') as f:
    inputs = pickle.load( f )

# Restart from CNT 32
inputs['vmec']['filename'] = 'inputs/vmec_inputs/input.cnt32'
inputs['cnt_coils']['geometry']['filename'] = 'inputs/coil_inputs/biotsavart_cnt32.json'

#inputs['cnt_coils']['target']['IL_maxR_weight'] = Weight(10)
#inputs['cnt_coils']['target']['IL_maxZ_weight'] = Weight(10)
#inputs['cnt_coils']['dofs']['R00_free'] = True

#inputs['cnt_coils']['target']['arclength_weight'] = Weight(1)
#inputs['cnt_coils']['target']['IL_length'] = 4.5

#inputs['vmec']['target']['aspect_ratio_weight'] = Weight(0)
#inputs['vmec']['target']['volume'] = 0.2
#inputs['vmec']['target']['volume_weight'] = Weight( 1E3 )
#inputs['vmec']['target']['volume_constraint_type'] = 'min'

inputs['cnt_coils']['target']['IL_vessel_threshold'] = -0.16
inputs['cnt_coils']['target']['IL_vessel_weight'] = 1E6

#inputs['numerics']['MAXITER_single_stage'] = 1E3

#inputs['coils_objective_weight'] = Weight(inputs['coils_objective_weight'].value / 10)
#inputs['vmec']['target']['iota_weight'] = Weight(inputs['vmec']['target']['iota_weight'].value / 10)
#inputs['vmec']['target']['volume_weight'] =  Weight(inputs['vmec']['target']['volume_weight'].value / 10)
#inputs['vmec']['target']['aspect_ratio_weight'] = Weight(inputs['vmec']['target']['aspect_ratio_weight'].value / 10)

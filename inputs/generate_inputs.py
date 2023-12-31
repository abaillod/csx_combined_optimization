import os
import single_stage_cnt32
import numpy as np
from simsopt.objectives import Weight
import pickle
import os

inputs = single_stage_cnt32.inputs
input_dir = 'cnt32_staged'
os.makedirs( input_dir, exist_ok=True )

res = 3

# Stage approach
last_dir = ''
for ii, morn in enumerate([1,2,3,4,5]):
    # Set parameter space:
    inputs['vmec']['dofs']['mpol'] = morn
    inputs['vmec']['dofs']['mpol'] = morn

    # Increase VMEC resolution accordingly
    inputs['vmec']['internal_mpol'] = morn+2
    inputs['vmec']['internal_ntor'] = morn+2

    # Set output directory
    inputs['directory'] = f'runs/cnt32_staged_M=N={morn:d}'

    # If not first run, depend on previous runs
    if ii>0:
        inputs['vmec']['filename'] = os.path.join(last_dir, 'input.final')
        inputs['cnt_coils']['geometry']['filename'] = os.path.join(last_dir, 'coils/bs_output.json')


    # Save input
    with open(os.path.join(input_dir, f'input_cnt32_{ii:d}.pckl'), 'wb') as f:
        pickle.dump( inputs, f )

    # Update last directory
    last_dir = inputs['directory']

    










#coil_weight = np.linspace(1, 1e2, 5)
#coil_constraint_weight = np.linspace(-5,0,6)

#counter=0
#for cw in coil_weight:
#    for ccw in coil_constraint_weight:
#        counter+=1
#        inputs['vmec']['dofs']['internal_mpol'] = res
#        inputs['vmec']['dofs']['internal_ntor'] = res
#
#        inputs['directory'] = f'runs/cnt32_improve_qs_mpol=ntor={res}_cweight={cw}_wcurve={ccw}'
#        inputs['coils_objective_weight'] = Weight(cw)
#        inputs['cnt_coils']['target']['IL_length_weight'] = Weight(10**(-ccw))
#        inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(10**(-ccw))
#        inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(10**(-ccw))
#        inputs['cnt_coils']['target']['PF_current_weight'] = Weight(10**(-ccw))
#
#        inputs['vmec']['filename'] = f'runs/cnt32_improve_qs_mpol=ntor={res-1}_cweight={cw}_wcurve={ccw}/input.final'
#        inputs['cnt_coils']['geometry']['filename'] = f'cnt32_improve_qs_mpol=ntor={res-1}_cweight={cw}_wcurve={ccw}/coils/bs_output.json'
#
#        with open(os.path.join(f'scan_mpol=ntor={res}', f'cnt32_{counter}.pckl'), 'wb') as f:
#            pickle.dump(inputs, f)

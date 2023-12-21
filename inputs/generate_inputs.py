import os
import single_stage_cnt32
import numpy as np
from simsopt.objectives import Weight
import pickle

inputs = single_stage_cnt32.inputs

res = 3

inputs['vmec']['dofs']['internal_mpol'] = res
inputs['vmec']['dofs']['internal_ntor'] = res


coil_weight = np.linspace(1, 1e2, 5)
coil_constraint_weight = np.linspace(-5,0,6)

counter=0
for cw in coil_weight:
    for ccw in coil_constraint_weight:
        counter+=1
        inputs['directory'] = f'runs/cnt32_improve_qs_mpol=ntor={res}_cweight={cw}_wcurve={ccw}'
        inputs['coils_objective_weight'] = Weight(cw)
        inputs['cnt_coils']['target']['IL_length_weight'] = Weight(10**(-ccw))
        inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(10**(-ccw))
        inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(10**(-ccw))
        inputs['cnt_coils']['target']['PF_current_weight'] = Weight(10**(-ccw))

        inputs['vmec']['filename'] = f'runs/cnt32_improve_qs_mpol=ntor={res-1}_cweight={cw}_wcurve={ccw}/input.final'
        inputs['cnt_coils']['geometry']['filename'] = f'cnt32_improve_qs_mpol=ntor={res-1}_cweight={cw}_wcurve={ccw}/coils/bs_output.json'

        with open(os.path.join(f'scan_mpol=ntor={res}', f'cnt32_{counter}.pckl'), 'wb') as f:
            pickle.dump(inputs, f)

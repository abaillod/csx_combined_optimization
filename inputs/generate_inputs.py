import os
import single_stage_cnt32
import numpy as np
from simsopt.objectives import Weight
import pickle

inputs = single_stage_cnt32.inputs

inputs['vmec']['dofs']['mpol'] = 1
inputs['vmec']['dofs']['ntor'] = 1


coil_weight = np.linspace(1, 1e2, 5)
coil_constraint_weight = np.linspace(-5,0,6)

counter=0
for cw in coil_weight:
    for ccw in coil_constraint_weight:
        counter+=1
        inputs['directory'] = f'runs/cnt32_improve_qs_mpol=ntor=1_cweight={cw}_wcurve={ccw}'
        inputs['coils_objective_weight'] = Weight(cw)
        inputs['cnt_coils']['target']['IL_length_weight'] = Weight(ccw)
        inputs['cnt_coils']['target']['IL_msc_weight'] = Weight(ccw)
        inputs['cnt_coils']['target']['IL_maxc_weight'] = Weight(ccw)
        inputs['cnt_coils']['target']['PF_current_weight'] = Weight(ccw)

        with open(os.path.join('scan_mpol=ntor=1', f'cnt32_{counter}.pckl'), 'wb') as f:
            pickle.dump(inputs, f)

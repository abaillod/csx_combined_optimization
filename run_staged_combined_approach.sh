#!/bin/sh

last_morn=-1
for morn in 2 3
do
	directory=combined_approach_staged_$morn
	mkdir runs/$directory

	cp inputs/qs_boundary_no_coil_objective.py runs/$directory/input.py
	echo "inputs['directory']='runs/$directory'" >> runs/$directory/input.py
	echo "inputs['vmec']['dofs']['mpol'] = $morn" >> runs/$directory/input.py
	echo "inputs['vmec']['dofs']['ntor'] = $morn" >> runs/$directory/input.py

	if [ $last_morn -eq -1 ]
	then
		echo "inputs['vmec']['filename'] = 'input.vacuum_cssc_scaled'" >> runs/$directory/input.py
		jid=$(sbatch --parsable run_combined_approach.sb runs.$directory.input)
		echo "Started run $jid, with no dependencies. Using mpol=ntor=$morn."
	else
		last_directory=combined_approach_staged_$last_morn
		echo "inputs['vmec']['filename'] = 'runs/$last_directory/input.final'" >> runs/$directory/input.py
		last_jid=$jid
		jid=$(sbatch --parsable --dependency=afterany:$jid run_combined_approach.sb  runs.$directory.input)
		echo "Started run $jid, with dependencies on run $last_jid. Using mpol=ntor=$morn."
	fi	
	last_morn=$morn
done


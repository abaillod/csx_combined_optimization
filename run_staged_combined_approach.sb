#!/bin/sh

inputfile="'input.vacuum_cssc_scaled'"
date=$(date '+%Y-%m-%dT%H:%M')

last_morn=-1
for morn in 1 2 3 4 5 6 
do
	directory=$date\_mpol\=ntor\=$morn
	last_directory=-1
        let vmec_mpol=$morn+2
	mkdir runs/$directory

	cp inputs/qs_boundary_no_coil_objective.py runs/$directory/input.py
	echo "inputs['directory']='runs/$directory'" >> runs/$directory/input.py
	echo "inputs['vmec']['dofs']['mpol'] = $morn" >> runs/$directory/input.py
	echo "inputs['vmec']['dofs']['ntor'] = $morn" >> runs/$directory/input.py
	echo "inputs['vmec']['internal_mpol'] = $vmec_mpol" >> runs/$directory/input.py

	if [ $last_morn -eq -1 ]
	then
		echo "inputs['vmec']['filename'] = $inputfile" >> runs/$directory/input.py
		jid=$(sbatch --parsable --output=runs/$directory/slurm_output.out --error=runs/$directory/slurm_error.out run_combined_approach.sb runs.$directory.input)
		echo "Started run $jid, with no dependencies. Using mpol=ntor=$morn."
	else
		echo "inputs['vmec']['filename'] = 'runs/$last_directory/input.final'" >> runs/$directory/input.py
		last_jid=$jid
		jid=$(sbatch --parsable --output=runs/$directory/slurm_output.out --error=runs/$directory/slurm_error.out --dependency=afterany:$jid run_combined_approach.sb  runs.$directory.input)
		echo "Started run $jid, with dependencies on run $last_jid. Using mpol=ntor=$morn."
	fi	
	last_morn=$morn
	last_directory=$directory
done


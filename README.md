# CSX combined optimization
This repository uses the [simsopt framework](https://github.com/hiddenSymmetries/simsopt) and the combined approach technique described by [R. Jorge et.al.](https://iopscience.iop.org/article/10.1088/1361-6587/acd957) 
to optimize tentative designs of the Columbia Stellarator eXperiment (CSX), which will eventually replace the existing [Columbia Nonneutral Torus (CNT)](http://sites.apam.columbia.edu/CNT/index.htm)

![bestview](https://github.com/abaillod/csx_combined_optimization/assets/45510759/74bc1225-0b45-42d6-8d0c-3010ae5ad3cc)


## Installation
No installation is required (except simsopt and VMEC2000)

## Running the code
The main script is `combined_csx_optimization_with_windowpane_coils.py`. It can be run locally by creating an input file in the `./inputs` directory - just copy-paste the `inputs/standard_input.py` file into a new
file, and set each input to the value you desire. Then, run the code locally (not recommended) with:
```
python combined_csx_optimization_with_windowpane_coils.py name_of_your_input
```
To run the code on the Ginsburg cluster, simply use the sbatch script `run_combined_approach.sb` and do
```
sbatch run_combined_approach.sb name_of_your_input
```

## Analysing the output
To be completed...

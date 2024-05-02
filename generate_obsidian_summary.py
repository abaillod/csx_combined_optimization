import booz_xform as bx
from simsopt.mhd.vmec import Vmec
import argparse
import os
from simsopt._core.optimizable import load, save
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field import BiotSavart
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo import CurveLength
import pickle


# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--opt", dest="opt", default=None)
parser.add_argument("--MN", dest="MN", default=None)

# Prepare args
args = parser.parse_args()

pp = f'runs/opt_{args.opt}/M=N={args.MN}'
name = f'OPT_{args.opt}_M=N={args.MN}'

this_path = os.path.join(os.getcwd(), pp)
figure_path = os.path.join( this_path, 'figure')
obsidian_path = '/Users/antoinebaillod/Library/Mobile Documents/iCloud~md~obsidian/Documents/Research/CORE/RUNS'
os.makedirs(figure_path, exist_ok=True)

os.chdir(this_path)

v = Vmec('input.final')
bs = load(os.path.join(this_path, 'coils/bs_output.json'))
v.run()

# Run boozer
b = bx.Booz_xform()
b.read_wout(v.output_file)
b.mboz = 48
b.nboz = 48
b.run()
b.write_boozmn("boozmn.nc")
s = b.s_b

# Evaluate figures of merit
bmnc = b.bmnc_b
xm = b.xm_b
xn = b.xn_b
f = np.sqrt(np.sum(bmnc[xn!=0,:]**2,axis=0)/np.sum(bmnc**2,axis=0))
fqs = np.mean(f)
mean_iota = np.mean(b.iota)
volume = v.boundary.volume()
major_radius = v.boundary.major_radius()
minor_radius = v.boundary.minor_radius()
aspect_ratio = v.boundary.aspect_ratio()
il_current = np.abs(bs.coils[0].current.get_value())
pf_current = np.abs(bs.coils[2].current.get_value())
ll = CurveLength( bs.coils[0].curve )
il_length = ll.J()


fname = name + '.md'
with open( os.path.join(obsidian_path, fname), 'w') as file:
    file.write("---\n")
    file.write("tags:\n")
    file.write("  - csx\n")
    file.write("  - vmec_combined_approach\n")
    file.write("  - optimization/run\n")
    file.write("aliases:\n")
    file.write(f"  - OPT_{args.opt}\n")
    file.write(f'name: "{name}"\n')
    file.write(f'mean iota: "{mean_iota:.2E}"\n')
    file.write(f'fQS: "{fqs:.2E}"\n')
    file.write(f'volume: "{volume:.2E}"\n')
    file.write(f'rmaj: "{major_radius:.2E}"\n')
    file.write(f'rmin: "{minor_radius:.2E}"\n')
    file.write(f'aspect: "{aspect_ratio:.2E}"\n')
    file.write(f'IL current: "{il_current:.2E}"\n')
    file.write(f'PF current: "{pf_current:.2E}"\n')
    file.write(f'IL length: "{il_length:.2E}"\n')
    file.write("---\n\n\n")
    file.write("## Notes\n\n\n")
    file.write("## Inputs\n\n")
    file.write("```\n")
    with open( os.path.join(this_path, 'input.txt'), 'r' ) as input_file:
        for line in input_file:
            file.write(line)
    file.write("```\n\n\n")
    file.write("## Figures\n")
    file.write("### 3D plot of coils and surface\n")
    file.write("### Poincare section\n")
    file.write(f"![poincare](file:{os.path.join(figure_path, 'poincare.png')})\n")
    file.write("### |B| on plasma boundary\n")
    file.write(f"![modb](file:{os.path.join(figure_path, 'modB.png')})\n")
    file.write("### $\mathbf{B}\cdot\hat{n}$ error (normalized)\n")
    file.write(f"![normal_field_error](file:{os.path.join(figure_path, 'normal_field_error.png')})\n")
    file.write("### Profiles\n")
    file.write(f"![fqs](file:{os.path.join(figure_path, 'fquasisymmetry.png')})\n")
    file.write(f"![iota](file:{os.path.join(figure_path, 'iota.png')})\n")



    



# with open( os.path.join(this_path, 'figures_of_merit.txt'), 'w' ) as file:
#           file.write(f'Mean iota = {mean_iota:.2E}\n')
#           file.write(f'f_quasisymmetry = {fqs:.2E}\n')
#           file.write(f'Volume = {volume:.2E}\n')
#           file.write(f'Major radius = {major_radius:.2E}\n')
#           file.write(f'Minor radius = {minor_radius:.2E}\n')
#           file.write(f'Aspect ratio = {aspect_ratio:.2E}\n')
#           file.write(f'IL coil current = {il_current:.2E}\n')
#           file.write(f'PF coil current = {pf_current:.2E}\n')
#           file.write(f'IL coil length = {il_length:.2E}\n')
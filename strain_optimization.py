from simsopt.geo.framedcurve import FramedCurveFrenet, FramedCurveCentroid
from simsopt.geo import CurveXYZFourier
from simsopt.configs import get_ncsx_data, get_hsx_data
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import gridToVTK
from simsopt.geo import CurveLength, FramedCurveTwist, CoilStrain, LPTorsionalStrainPenalty, LPBinormalCurvatureStrainPenalty, FrameRotation
from scipy.optimize import minimize
from simsopt._core import load
from simsopt.objectives import QuadraticPenalty

rot_order = 10
coils = load('boozer_surface_rogerio_opt_3.12_0_253_10.json').biotsavart.coils
curves = [coil.curve for coil in coils]
curve = curves[0]
curve.fix_all()

rotation = FrameRotation(curve.quadpoints, rot_order)

fc_centroid = FramedCurveCentroid(curve)
fc_frenet = FramedCurveFrenet(curve)
fc = FramedCurveCentroid(curve,rotation)

#rotation_frenet = FrameRotation(curve.quadpoints, rot_order)
#rotation_frenet.set('xc(0)',np.pi)
#fc_frenet = FramedCurveCentroid(curve,rotation_frenet)
#print(FramedCurveTwist(fc_frenet,rotation_frenet).J())

twist = FramedCurveTwist(fc)

cs = CoilStrain(fc, width=4e-3)

tor_threshold = 1e-3  # Threshold for strain parameters
cur_threshold = 1e-3
width = 4e-3

Jtor = LPTorsionalStrainPenalty(fc, p=2, threshold=tor_threshold)
Jbin = LPBinormalCurvatureStrainPenalty(
    fc, p=2, threshold=cur_threshold)
Jtwist = QuadraticPenalty(twist,0.5,'max')

JF = Jtor + 10*Jbin + Jtwist

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    outstr = f"Max torsional strain={np.max(cs.torsional_strain()):.1e}, Max curvature strain={np.max(cs.binormal_curvature_strain()):.1e}, Frame twist={twist.J()}, Jtwist={Jtwist.J()}"
    print(outstr)
    return J, grad

f = fun
dofs = JF.x

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)

print(res.message)

JF.save(filename='strainopt.json')

fc_opt = fc

for i, fc in enumerate([fc_opt, fc_centroid, fc_frenet]):

	t, n, b = fc.rotated_frame()
	gamma = fc.curve.gamma()
	x = gamma[:,0]
	y = gamma[:,1]
	z = gamma[:,2]
	x = np.concatenate([x, [x[0]]])
	y = np.concatenate([y, [y[0]]])
	z = np.concatenate([z, [z[0]]])
	vx = b[:,0]
	vy = b[:,1]
	vz = b[:,2]
	vx = np.concatenate([vx, [vx[0]]])
	vy = np.concatenate([vy, [vy[0]]])
	vz = np.concatenate([vz, [vz[0]]])

	dist = 0.01*CurveLength(fc.curve).J()
	width = np.linspace(0,dist,100)
	x = x[:,None]+width[None,:]*vx[:,None]
	y = y[:,None]+width[None,:]*vy[:,None]
	z = z[:,None]+width[None,:]*vz[:,None]

	x = x.reshape((1, len(vx), len(width))).copy()
	y = y.reshape((1, len(vx), len(width))).copy()
	z = z.reshape((1, len(vx), len(width))).copy()

	twist = fc.frame_twist() - fc_centroid.frame_twist()
	twist = np.concatenate([twist, [twist[0]]])
	filename = 'curvewidth_notwist_'+str(i)+'_centroid'
	data = np.ones((np.shape(x)[1],np.shape(x)[2]))
	data[:,:] = twist[:,None]*np.ones_like(x)

	data = {"twist": (data)[:, :, None]}
	gridToVTK(str(filename), np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z), pointData=data)
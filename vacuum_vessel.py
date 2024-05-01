""" Define the CSX vacuum vessel boundary

We will use this boundary to construct constraints for IL coils
and to support the WP coils
"""

from simsopt.geo import Surface
from jax import grad
import jax.numpy as jnp
np = jnp

from simsopt.geo.jit import jit
from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

def gamma( tarr, zarr, params ):        
    barrel_OD = params['barrel_OD']       
    barrel_thickness = params['barrel_thickness']
    barrel_height = params['barrel_height']
    flange_gap = params['flange_gap']
    
    # Elliptical head radii
    DR = params['DR'] 
    KR = params['KR']
    
    head_thickness = params['head_thickness']
    
    flange_height = params['flange_height']
    KR_height_top = params['KR_height_top']
    
    KR_height_bottom = params['KR_height_bottom']
    KR_centerR = params['KR_centerR']
    DR_height_top = params['DR_height_top']
    DR_height_bottom = params['DR_height_bottom']
    math = np
    
    nz = zarr.size
    nt = tarr.size
    
    rarr = np.zeros((nz,))
    for iz in range(nz):
        z = zarr[iz]
        #DR
        if (z>=KR_height_top): # ???????????
            rarr = rarr.at[iz].set( math.sqrt((DR-head_thickness)**2-(z-DR_height_top+(DR-head_thickness))**2) )
        #KR
        if (z>=flange_height and z<KR_height_top): # Second part of cap?
            rarr = rarr.at[iz].set( math.sqrt((KR-head_thickness)**2-(z-flange_height)**2)+KR_centerR )
        #Barrel
        if (z>=-barrel_height/2 and z<flange_height): # cylindrical barrel
            rarr = rarr.at[iz].set( barrel_OD/2 - barrel_thickness )
        #KR
        if (z>=KR_height_bottom and z<-barrel_height/2):
            rarr = rarr.at[iz].set( math.sqrt((KR-head_thickness)**2-(z+barrel_height/2)**2)+KR_centerR )
        #DR
        if (z<KR_height_bottom):
            rarr = rarr.at[iz].set( math.sqrt((DR-head_thickness)**2-(z-DR_height_bottom-(DR-head_thickness))**2) )

    
    
    out = np.zeros((nt*nz, 3))
    counter=-1
    for it in range(nt):
        for iz in range(nz):
            counter+=1
            out = out.at[counter,0].set( rarr[iz] * np.sin( tarr[it] ) )
            out = out.at[counter,1].set( zarr[iz] )
            out = out.at[counter,2].set( rarr[iz] * np.cos( tarr[it] ) )
    
    return out


def normal( tarr, zarr, params ): 
    barrel_OD = params['barrel_OD']       
    barrel_thickness = params['barrel_thickness']
    barrel_height = params['barrel_height']
    flange_gap = params['flange_gap']
    
    # Elliptical head radii
    DR = params['DR'] 
    KR = params['KR']
    
    head_thickness = params['head_thickness']
    
    flange_height = params['flange_height']
    KR_height_top = params['KR_height_top']
    
    KR_height_bottom = params['KR_height_bottom']
    KR_centerR = params['KR_centerR']
    DR_height_top = params['DR_height_top']
    DR_height_bottom = params['DR_height_bottom']
    math = np
    
    nz = zarr.size
    nt = tarr.size
    
    rarr = np.zeros((nz,))
    drdz = np.zeros((nz,))
    # Sign of derivatives is chosen such that normal vector is pointing outwards - this is not
    # mathematically equal to the derivative
    for iz in range(nz):
        z = zarr[iz]
        #DR
        if (z>=KR_height_top): # ???????????
            rarr = rarr.at[iz].set( math.sqrt((DR-head_thickness)**2-(z-DR_height_top+(DR-head_thickness))**2) )
            drdz = drdz.at[iz].set( -(z-DR_height_top+(DR-head_thickness)) / rarr[iz] )
        #KR
        if (z>=flange_height and z<KR_height_top): # Second part of cap?
            rarr = rarr.at[iz].set( math.sqrt((KR-head_thickness)**2-(z-flange_height)**2)+KR_centerR )
            drdz = drdz.at[iz].set( (z-flange_height) / (KR_centerR - rarr[iz]) )
        #Barrel
        if (z>=-barrel_height/2 and z<flange_height): # cylindrical barrel
            rarr = rarr.at[iz].set( barrel_OD/2 - barrel_thickness )
        #KR
        if (z>=KR_height_bottom and z<-barrel_height/2):
            rarr = rarr.at[iz].set( math.sqrt((KR-head_thickness)**2-(z+barrel_height/2)**2)+KR_centerR )
            drdz = drdz.at[iz].set( (z+barrel_height/2) / (KR_centerR - rarr[iz]) )
        #DR
        if (z<KR_height_bottom):
            rarr = rarr.at[iz].set( math.sqrt((DR-head_thickness)**2-(z-DR_height_bottom-(DR-head_thickness))**2) )
            drdz = drdz.at[iz].set( -(z-DR_height_bottom-(DR-head_thickness)) / rarr[iz] )


    dtheta = np.zeros((nt*nz, 3))
    dz = np.zeros((nt*nz, 3))
    dz = dz.at[:,1].set( 1 )
    counter=-1
    for it in range(nt):
        for iz in range(nz):
            counter+=1
            dtheta = dtheta.at[counter,0].set( rarr[iz] * np.cos( tarr[it] ) )
            dtheta = dtheta.at[counter,2].set(-rarr[iz] * np.sin( tarr[it] ) )
            
            dz = dz.at[counter,0].set( drdz[iz] * np.sin( tarr[it] ) )
            dz = dz.at[counter,2].set( drdz[iz] * np.cos( tarr[it] ) )
    
    return np.cross( dtheta, dz )
    


class CSX_VacuumVessel:
    def __init__(self, ntheta=128, nz = 152, scale=1):
        f = 0.0254 # inch to meters
        epsilon = 0.1
        self.scale = scale

        self.nz = nz
        self.nt = ntheta
        
        # Vessel Params
        params = dict()
        params['barrel_OD'] = 60    
        params['barrel_thickness'] = 0.25
        params['barrel_height'] = 38.81
        params['flange_gap'] = 1.5
        
        # Elliptical head radii
        params['DR'] = params['barrel_OD'] * 0.9
        params['KR'] = params['barrel_OD'] * 0.173
        params['head_thickness'] = 0.3125
        params['flange_height'] = params['flange_gap'] + params['barrel_height']/2
        params['KR_height_top'] = 8.991611 + params['flange_gap'] + params['barrel_height']/2
        params['KR_height_bottom'] = -8.991611 - params['barrel_height']/2
        params['KR_centerR'] = 19.62
        params['DR_height_top'] = params['flange_height'] + 14.729061
        params['DR_height_bottom'] = (-38.81/2-14.729061)
        self.params = params
        
        self.quadpoints_theta = np.linspace(0, 2*np.pi, ntheta)
        self.quadpoints_z = np.linspace(params['DR_height_bottom']+epsilon, params['DR_height_top']-epsilon, nz)

        # Only evaluate it once
        self.gamma_stack = gamma( self.quadpoints_theta, self.quadpoints_z, self.params ) * f
        self.normal_stack = normal( self.quadpoints_theta, self.quadpoints_z, self.params )
        self.unit_normal_stack = np.einsum( 'ij,i->ij', self.normal_stack, 1./np.linalg.norm(self.normal_stack, axis=1) )


    def gamma(self):
        return self.gamma_stack.reshape((self.nz,self.nt,3)) * self.scale

    def normal(self):
        return self.normal_stack

    def unitnormal(self):
        return self.unit_normal_stack

def signed_distance_from_surface(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma().reshape((-1, 3))
    mins = jnp.argmin( jnp.sum((gammas[:, None, :] - xyz[None, :, :])**2, axis=2), axis=0 )

    n = surface.unitnormal().reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]

    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = np.sum((xyz-gammamins) * nmins, axis=1)

    a_point_in_the_surface = np.mean(surface.gamma()[0, :, :], axis=0)
    sign_of_interiorpoint = np.sign(np.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))

    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists




def ws_distance_pure(gammac, lc, surface, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    ns = surface.normal().reshape((-1, 3))
    gammas = surface.gamma().reshape((-1,3))
    
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


def minimum_distance(gammac, surface):
    """
    This function returns the minimum distance between a curve and a surface
    """
    ns = surface.normal().reshape((-1, 3))
    gammas = surface.gamma().reshape((-1,3))
    
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    return np.min(dists)

class VesselConstraint(Optimizable):
    r"""Used to constrain coils to remain on a surface
    
    Computed
    .. math:
        J = \sum_{i=1}^{\text{num_coils}} d_i

    where
    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. This penalty is zero when all points are on the surface.
    """
    def __init__(self, curves, surface, maximum_distance):
        self.curves = curves
        self.surface = surface
        self.maximum_distance = maximum_distance
        #gammas = self.surface.gamma().reshape((-1,3))
        #ns = self.surface.normal().reshape((-1, 3))

        self.J_jax = jit(lambda gammac, lc: ws_distance_pure(gammac, lc, self.surface, self.maximum_distance))
        self.thisgrad0 = jit(lambda gammac, lc: grad(self.J_jax, argnums=0)(gammac, lc))
        self.thisgrad1 = jit(lambda gammac, lc: grad(self.J_jax, argnums=1)(gammac, lc))
        
        super().__init__(depends_on=curves)    

    def minimum_distances(self):
        res = []
        for c in self.curves:
            gammac = c.gamma()
            res.append(minimum_distance(gammac, self.surface))

        return res
    
    def J(self):
        """
        This returns the value of the quantity.
        """
        res = 0
        for c in self.curves:
            gammac = c.gamma()
            lc = c.gammadash()
            res += self.J_jax(gammac, lc)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, c in enumerate(self.curves):
            gammac = c.gamma()
            lc = c.gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + \
               self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))
              ]
        return sum(res)

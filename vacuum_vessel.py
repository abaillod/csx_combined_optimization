""" Define the CSX vacuum vessel boundary

We will use this boundary to construct constraints for IL coils
and to support the WP coils
"""

from simsopt.geo import Surface
import numpy as np

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
            rarr[iz] = math.sqrt((DR-head_thickness)**2-(z-DR_height_top+(DR-head_thickness))**2)
        #KR
        if (z>=flange_height and z<KR_height_top): # Second part of cap?
            rarr[iz] = math.sqrt((KR-head_thickness)**2-(z-flange_height)**2)+KR_centerR
        #Barrel
        if (z>=-barrel_height/2 and z<flange_height): # cylindrical barrel
            rarr[iz] = barrel_OD/2 - barrel_thickness 
        #KR
        if (z>=KR_height_bottom and z<-barrel_height/2):
            rarr[iz] = math.sqrt((KR-head_thickness)**2-(z+barrel_height/2)**2)+KR_centerR              
        #DR
        if (z<KR_height_bottom):
            rarr[iz] = math.sqrt((DR-head_thickness)**2-(z-DR_height_bottom-(DR-head_thickness))**2)

    
    
    out = np.zeros((nt*nz, 3))
    counter=-1
    for it in range(nt):
        for iz in range(nz):
            counter+=1
            out[counter,0] = rarr[iz] * np.sin( tarr[it] ) 
            out[counter,1] = zarr[iz] 
            out[counter,2] = rarr[iz] * np.cos( tarr[it] ) 
    
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
            rarr[iz] = math.sqrt((DR-head_thickness)**2-(z-DR_height_top+(DR-head_thickness))**2)
            drdz[iz] = -(z-DR_height_top+(DR-head_thickness)) / rarr[iz]
        #KR
        if (z>=flange_height and z<KR_height_top): # Second part of cap?
            rarr[iz] = math.sqrt((KR-head_thickness)**2-(z-flange_height)**2)+KR_centerR
            drdz[iz] = (z-flange_height) / (KR_centerR - rarr[iz])
        #Barrel
        if (z>=-barrel_height/2 and z<flange_height): # cylindrical barrel
            rarr[iz] = barrel_OD/2 - barrel_thickness 
            drdz[iz] = 0
        #KR
        if (z>=KR_height_bottom and z<-barrel_height/2):
            rarr[iz] = math.sqrt((KR-head_thickness)**2-(z+barrel_height/2)**2)+KR_centerR              
            drdz[iz] = (z+barrel_height/2) / (KR_centerR - rarr[iz])
        #DR
        if (z<KR_height_bottom):
            rarr[iz] = math.sqrt((DR-head_thickness)**2-(z-DR_height_bottom-(DR-head_thickness))**2)
            drdz[iz] = -(z-DR_height_bottom-(DR-head_thickness)) / rarr[iz]


    dtheta = np.zeros((nt*nz, 3))
    dz = np.zeros((nt*nz, 3))
    dz[:,1] = 1
    counter=-1
    for it in range(nt):
        for iz in range(nz):
            counter+=1
            dtheta[counter,0] = rarr[iz] * np.cos( tarr[it] )
            dtheta[counter,2] =-rarr[iz] * np.sin( tarr[it] )
            
            dz[counter,0] = drdz[iz] * np.sin( tarr[it] )
            dz[counter,2] = drdz[iz] * np.cos( tarr[it] )
    
    normal = np.cross( dtheta, dz )
    norm   = np.linalg.norm( normal, axis=1 )    
    
    return np.einsum('ij,i->ij', normal, 1./norm)
    


class CSX_VacuumVessel:
    def __init__(self, ntheta=64, nz = 32, scale=1):
        f = 0.0254 # inch to meters
        epsilon = 0.1
        self.scale = scale
        
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
        self.unit_normal_stack = normal( self.quadpoints_theta, self.quadpoints_z, self.params )


    def gamma(self):
        return self.gamma_stack * self.scale

    def normal(self):
        return self.unit_normal_stack
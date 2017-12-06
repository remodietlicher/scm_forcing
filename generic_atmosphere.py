#!/usr/bin/python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------- HUMIDITY UTILS -----------------------------
eps = 0.622   # Rd/Rv
tmlt = 273.15 # melting temp
r_d = 287.058 # specific gas constant
grav = 9.81

def murphy_koop_esw(T):
    a = 54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
    b = np.tanh(0.0415*(T-218.8))
    c = 53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T

    return np.exp(a+b*c)

def murphy_koop_esi(T):
    a = 9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T
    
    return np.minimum(np.exp(a), murphy_koop_esw(T))

def p_es_rh_to_q(p, es, rh):
    e = rh*es
    w = e*eps/(p-e+eps*e)
    return w/(w+1)

def p_e_t_to_r(p, e, t):
    x = e/p
    tvt = 1/(1-x*(1-eps))
    r = x*eps*tvt
    return r    

def p_es_q_to_rh(p, es, q):
    x = es/p
    qs = x*eps/(1-x*(1-eps))
    return q/qs

# -------------------------- PRIVATE ------------------------------------
def find_nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_lower_index(array, value):
    idx = np.abs(array-value).argmin()
    lcorr = (array-value)[idx] > 0
    if(lcorr):
        idx = idx-1
    return idx

def find_upper_index(array, value):
    idx = np.abs(array-value).argmin()
    lcorr = (array-value)[idx] > 0
    if(lcorr):
        idx = idx+1
    return idx


# -------------------------- PUBLIC -------------------------------------

class generic_atmosphere:
    # initialize the atmosphere to the profiles given by p, t and q
    # data: An empty data file (requires write access)
    # 1D ARRAYS:
    # p   : pressure levels (Pa)
    # t   : temperature at level (K)
    # q   : relative humidity at level [0,1]
    def __init__(self, data, p, t, rh):
        # store data
        self.data = data
        # store time
        self.time = data.variables['time'][:]

        # initialize pressure levels
        # --------------------------
        vct_a = data.variables['vct_a'][:]
        vct_b = data.variables['vct_b'][:]
        aps = data.variables['aps'][:]

        nt, = self.time.shape
        nlevp1, = vct_a.shape
        nlev = nlevp1-1

        mlev = np.zeros((nt, nlev))
        ilev = np.zeros((nt, nlevp1))
        # calculate pressure levels at interfaces
        for i in range(nt):
            ilev[i,:] = vct_a + vct_b*aps[i]
        # calculate pressure levels at midpoint
        mlev[:,:] = ilev[:,:nlev] + 0.5*np.diff(ilev, axis=1)
        # store pressure level at midpoint
        self.mlev = mlev
        self.pmin = np.min(mlev)
        self.pmax = np.max(mlev)

        # set the initial profiles
        # ------------------------
        # interpolate the t and rh profiles
        f_t = interp1d(p, t, kind='linear')
        f_rh = interp1d(p, rh, kind='linear')

        # map t and rh to model levels
        self.m_t = f_t(self.mlev)
        self.m_rh = f_rh(self.mlev)

        # calculate saturation pressure w.r.t. liquid water
        m_esw = murphy_koop_esw(self.m_t)

        # calculate humidity mixing ratio
        m_q = p_es_rh_to_q(self.mlev, m_esw, self.m_rh)

        # write profiles to file
        self.data.variables['t'][:] = self.m_t
        self.data.variables['q'][:] = m_q
        return

    # set a layer for the variable 'name' to 'val'
    # p0  : pressure at the bottom of the layer (Pa)
    # p1  : pressure at the top of the layer (Pa)
    # rh  : value for each level (unit of variable)
    # name: name of the variable
    def add_rh_layer(self, p0, p1, rh):
        # interpolate additional rh profile
        p  = np.array([0, p0-1, p0, p1, p1+1, 120000])
        rh = np.array([0, 0, rh, rh, 0, 0])
        f_rh = interp1d(p, rh, kind='linear')

        # wrap with initial profile and map to model level
        self.m_rh = np.maximum(f_rh(self.mlev),self.m_rh)

        # calculate saturation pressure w.r.t. liquid water
        m_esw = murphy_koop_esw(self.m_t)

        # calculate humidity mixing ratio
        m_q = p_es_rh_to_q(self.mlev, m_esw, self.m_rh)

        # write profile to file
        self.data.variables['q'][:] = np.maximum(m_q, self.data.variables['q'][:])

        return

    # p0  : pressure at the bottom of the layer (Pa)
    # p1  : pressure at the top of the layer (Pa)
    # rh  : value for each level (unit of variable)
    def add_rhi_layer(self, p0, p1, rhi):
        # interpolate additional rhi profile
        p  = np.array([0, p0-1, p0, p1, p1+1, 120000])
        rhi = np.array([0, 0, rhi, rhi, 0, 0])
        f_rhi = interp1d(p, rhi, kind='linear')

        # get an array of rhi
        m_rhi = f_rhi(self.mlev)

        # calculate saturation pressure w.r.t. liquid water
        m_esi = murphy_koop_esi(self.m_t)
        m_esw = murphy_koop_esw(self.m_t)

        # calculate humidity mixing ratio
        m_q = p_es_rh_to_q(self.mlev, m_esi, m_rhi)

        # store corresponding rhw
        m_rhw = p_es_q_to_rh(self.mlev, m_esw, m_q)
        self.m_rh = np.maximum(m_rhw,self.m_rh)

        # write profile to file
        self.data.variables['q'][:] = np.maximum(m_q, self.data.variables['q'][:])

        return

    # set a layer for the variable 'name' to 'val'
    # p0  : pressure at the bottom of the layer (Pa)
    # p1  : pressure at the top of the layer (Pa)
    # val : value for the variable to be set (unit of variable)
    # name: name of the variable
    def add_layer(self, p0, p1, val, name):
        # interpolate additional rhi profile
        p  = np.array([self.pmin, p0-1, p0, p1, p1+1, self.pmax])
        val = np.array([0, 0, val, val, 0, 0])
        f_val = interp1d(p, val, kind='linear')

        # get an array of val
        m_val = f_val(self.mlev)

        # write profile to file
        self.data.variables[name][:] += m_val

        return

    # set a layer segment for the variable 'name' to 'val'
    # p0  : pressure at the bottom of the layer (Pa)
    # p1  : pressure at the top of the layer (Pa)
    # t0  : time of start of the layer (s)
    # t1  : time of end of the layer (s)
    # val : value for each level (unit of variable)
    # name: name of the variable 
    def add_layer_segment(self, p0, p1, t0, t1, val1, val2, name):
        # interpolate the profile
        p = np.array([self.pmin, p0-1, p0, p1, p1+1, self.pmax])
        v = np.array([0, 0, val1, val2, 0, 0])
        f_val = interp1d(p, v, kind='linear')

        # map to pressure levels
        m_val = f_val(self.mlev)

        # find time indices
        it0 = find_lower_index(self.time, t0)
        it1 = find_upper_index(self.time, t1)

        # write to file
        self.data.variables[name][it0:it1+1,:] += m_val[it0:it1+1,:]

        return

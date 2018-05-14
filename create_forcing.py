#!/usr/bin/python
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import numpy as np
import argparse
from generic_atmosphere import generic_atmosphere

# -------------------------- CONSTANTS ---------------------------------
caps = 101300.0
cts = 250
tmlt = 273.15

cvct_a = np.array([     0.        ,   2000.        ,   4000.        ,   6000.        ,
                     8000.        ,   9976.13671875,  11820.5390625 ,  13431.39453125,
                    14736.35546875,  15689.20703125,  16266.609375  ,  16465.00390625,
                    16297.62109375,  15791.59765625,  14985.26953125,  13925.51953125,
                    12665.29296875,  11261.23046875,   9771.40625   ,   8253.2109375 ,
                     6761.33984375,   5345.9140625 ,   4050.71777344,   2911.56933594,
                     1954.80517578,   1195.88989258,    638.14892578,    271.62646484,
                        72.06358337,      0.        ,      0.        ,      0.        ])

cvct_b = np.array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                     0.00000000e+00,   0.00000000e+00,   3.90858157e-04,
                     2.91970070e-03,   9.19413194e-03,   2.03191563e-02,
                     3.69748585e-02,   5.94876409e-02,   8.78949761e-02,
                     1.22003615e-01,   1.61441505e-01,   2.05703259e-01,
                     2.54188597e-01,   3.06235373e-01,   3.61145020e-01,
                     4.18202281e-01,   4.76688147e-01,   5.35886586e-01,
                     5.95084250e-01,   6.53564572e-01,   7.10594416e-01,
                     7.65405238e-01,   8.17166984e-01,   8.64955842e-01,
                     9.07715857e-01,   9.44213212e-01,   9.72985208e-01,
                     9.92281497e-01,   1.00000000e+00])

# -------------------------- 2D VARIABLES -------------------------------
ddt_div_atts      = {'units' : 'm s-2',
                     'long_name' : 'large scale u tendency'}
ddt_omega_atts    = {'units' : 'm s-2',
                     'long_name' : 'large scale u tendency'}
ddt_q_atts        = {'units' : 'kg s-1',
                     'long_name' : 'large scale q tendency'}
ddt_qi_atts        = {'units' : 'kg s-1',
                     'long_name' : 'large scale qi tendency'}
ddt_ni_atts        = {'units' : 's-1',
                     'long_name' : 'large scale ni tendency'}
ddt_qirim_atts     = {'units' : 'kg s-1',
                     'long_name' : 'large scale qirim tendency'}
ddt_birim_atts     = {'units' : 'm s-1',
                     'long_name' : 'large scale birim tendency'}
ddt_ql_atts        = {'units' : 'kg s-1',
                     'long_name' : 'large scale ql tendency'}
ddt_nl_atts        = {'units' : 's-1',
                     'long_name' : 'large scale nl tendency'}
ddt_nas_atts       = {'units' : 'm-3 s-1',
                     'long_name' : 'aerosol number tendency: AS'}
ddt_ms4as_atts     = {'units' : 'kg m-3 s-1',
                     'long_name' : 'SO4 tendency: AS'}
ddt_mduas_atts    = {'units' : 'kg m-3 s-1',
                     'long_name' : 'DU tendency: AS'}
ddt_t_atts        = {'units' : 'K s-1',
                     'long_name' : 'large scale T tendency'}
ddt_u_atts        = {'units' : 'm s-2',
                     'long_name' : 'large scale u tendency'}
ddt_v_atts        = {'units' : 'm s-2',
                     'long_name' : 'large scale v tendency'}
div_atts          = {'units' : '',
                     'long_name' : 'div'}
omega_atts        = {'units' : 'm s-2',
                     'long_name' : 'large scale subsidence'}
q_atts            = {'units' : 'kg kg-1',
                     'long_name' : 'specific humidity'}
qi_atts           = {'units' : 'kg kg-1',
                     'long_name' : 'ice mixing ratio'}
qirim_atts        = {'units' : 'kg kg-1',
                     'long_name' : 'rimed ice mixing ratio'}
ni_atts           = {'units' : 'kg-1',
                     'long_name' : 'ice number mixing ratio'}
birim_atts        = {'units' : 'm kg-1',
                     'long_name' : 'ice volume mixing ratio'}
ql_atts           = {'units' : 'kg kg-1',
                     'long_name' : 'liquid mixing ratio'}
nl_atts           = {'units' : 'kg-1',
                     'long_name' : 'liquid number mixing ratio'}
nas_atts          = {'units' : 'm-3',
                     'long_name' : 'aerosol number concentration: AS'}
ms4as_atts        = {'units' : 'kg m-3',
                     'long_name' : 'aerosol SO4 mass concentration: AS'}
mduas_atts        = {'units' : 'kg m-3',
                     'long_name' : 'aerosol DU mass concentration: AS'}
t_atts            = {'units' : 'K',
                     'long_name' : 'temperature'}
u_atts            = {'units' : 'm s-1',
                     'long_name' : 'horizontal velocity u'}
v_atts            = {'units' : 'm s-1',
                     'long_name' : 'horizontal velocity v'}
cessentials_2d = {'ddt_div':ddt_div_atts, 'ddt_omega':ddt_omega_atts, 'ddt_q':ddt_q_atts, 'ddt_qi':ddt_qi_atts, 'ddt_ql':ddt_qi_atts, 'ddt_qirim':ddt_qirim_atts, 'ddt_birim':ddt_birim_atts, 'ddt_ni':ddt_ni_atts, 'ddt_nas':ddt_nas_atts, 'ddt_mduas':ddt_mduas_atts, 'ddt_ms4as':ddt_ms4as_atts, 'ddt_t':ddt_t_atts, 'ddt_u':ddt_u_atts, 'ddt_v':ddt_v_atts, 'div':div_atts, 'omega':omega_atts, 'q':q_atts, 'xi':qi_atts, 'qirim':qirim_atts, 'ni':ni_atts, 'birim':birim_atts, 'xl':ql_atts, 'nl':nl_atts, 'nas':nas_atts, 'ms4as':ms4as_atts, 'mduas':mduas_atts, 't':t_atts, 'u':u_atts, 'v':v_atts}

# -------------------------- 1D VARIABLES -------------------------------
time_atts         = {'units' : 's',
                     'long_name' : 'time',
                     'start' : 2.4532835833333335e6} # date_start in julian date
aps_atts          = {'units' : 'Pa',
                     'long_name' : 'surface pressure'}
ts_atts           = {'units' : 'K',
                     'long_name' : 'surface temperature'}
vct_a_atts        = {'units' : '',
                     'long_name' : 'vertical-coordinate parameter set A'}
vct_b_atts        = {'units' : '',
                     'long_name' : 'vertical-coordinate parameter set B'}
lon_atts          = {'units' : 'degrees east',
                     'long_name' : 'longitude'}
lat_atts          = {'units' : 'degrees north',
                     'long_name' : 'latitude'}
date_atts         = {'units' : 'yyyymmddhh',
                     'long_name' : 'Gregorian start date in yyyymmddhh'}
slm_atts          = {'units' : 'mask 0,1',
                     'long_name' : 'land Sea mask'}
sn_atts           = {'units' : '',
                     'long_name' : ''}
lhf_atts          = {'units' : 'W m-2',
                     'long_name' : 'latent heat flux'}

# -------------------------- PARSER -------------------------------------

def get_parser():
    usage = """
            %(prog)s filename n_steps delta_t
            Example: %(prog)s day_forcing.nc 24 3600
                     This will initialize an empty (==0) forcing
                     file with 24 time steps, one every hour.
            """
    
    description = "Initialize an empty forcing file with n timesteps and timestep length dt"

    parser = argparse.ArgumentParser(description=description, usage=usage)

    parser.add_argument('filename', metavar='filename', type=str, help='name of the new forcing file')
    parser.add_argument('n_steps', metavar='n_steps', type=int, help='number of timesteps')
    parser.add_argument('delta_time', metavar='delta_time', type=float, help='time difference between steps')
    parser.add_argument('-r', dest='copyname', metavar='copyname', type=str, help='copy the first step of the given file')
    parser.add_argument('--std_atm', dest='std_atm', action='store_true', help='create the standard atmosphere forcing')
    parser.add_argument('--deep', dest='deep', action='store_true', help='create the deep cloud forcing')
    parser.add_argument('--warm', dest='warm', action='store_true', help='create the warm rain cloud forcing')
    parser.add_argument('--numerics', dest='numerics', action='store_true', help='create the test case for numerics')
    parser.add_argument('--infinite_cirrus', dest='infinite_cirrus', action='store_true', help='create the test case for infinite cirrus cloud generation')
    parser.set_defaults(deep=False)
    parser.set_defaults(std_atm=False)
    parser.set_defaults(warm=False)
    parser.set_defaults(numerics=False)
    parser.set_defaults(infinite_cirrus=False)

    return parser   


# -------------------------- MAIN ROUTINES ------------------------------


def create_file(name, n_steps, delta_time):
    print('creating file using name=%s, n_steps=%i, delta_time=%.1f'%(name, n_steps, delta_time))

    outfile = Dataset(name, 'w')
    
    outfile.createDimension(u'nlev', size=31)
    outfile.createDimension(u'time', size=n_steps+1)
    outfile.createDimension(u'nvclev', size=32)
    outfile.createDimension(u'lon', size=1)
    outfile.createDimension(u'lat', size=1)
    outfile.createDimension(u'date', size=1)
    outfile.createDimension(u'ncl5', size=1)
    outfile.createDimension(u'ncl6', size=2)

    for vname, atts in cessentials_2d.items():
        var = outfile.createVariable(vname, float, ('time', 'nlev'))
        var[:] = 0
        var.setncatts(atts)

    time = outfile.createVariable('time', float, ('time'))
    time.setncatts(time_atts)
    time[:] = np.linspace(0, delta_time*n_steps, n_steps+1)
    
    aps = outfile.createVariable('aps', float, ('time'))
    time.setncatts(aps_atts)
    aps[:] = caps*np.ones((n_steps+1))

    ts = outfile.createVariable('ts', float, ('time'))
    time.setncatts(ts_atts)
    ts[:] = cts*np.ones((n_steps+1))

    vct_a = outfile.createVariable('vct_a', float, ('nvclev'))
    vct_a.setncatts(vct_a_atts)
    vct_a[:] = cvct_a[:]

    vct_b = outfile.createVariable('vct_b', float, ('nvclev'))
    vct_b.setncatts(vct_b_atts)
    vct_b[:] = cvct_b[:]

    lon = outfile.createVariable('lon', float, ('lon'))
    lon.setncatts(lon_atts)
    lon[:] = 0

    lat = outfile.createVariable('lat', float, ('lat'))
    lat.setncatts(lat_atts)
    lat[:] = 0

    date = outfile.createVariable('date', float, ('date'))
    date.setncatts(date_atts)
    date[:] = 2004100502

    slm = outfile.createVariable('slm', float, ('ncl5'))
    slm.setncatts(slm_atts)
    slm[:] = 1

    sn = outfile.createVariable('sn', float, ('ncl6'))
    sn.setncatts(sn_atts)
    sn[:] = 0

    lhf = outfile.createVariable('lhf', float, ('time'))
    lhf.setncatts(lhf_atts)
    lhf[:] = 0

    outfile.sync()
    outfile.close()

def copy_from_file(filename, copyname):
    enh = 1
    outfile = Dataset(filename, 'a')
    copyfile = Dataset(copyname, 'r')
    for i in np.arange(0,np.floor((args.n_steps+1)/3.0)):
        outfile.variables['ddt_t'][i,:] = enh*copyfile.variables['ddt_t'][0,:]
        outfile.variables['ddt_q'][i,:] = enh*copyfile.variables['ddt_q'][0,:]
    for i in np.arange(np.floor((args.n_steps+1)/3.0), np.floor(2*(args.n_steps+1)/3.0)):
        outfile.variables['ddt_q'][i,:] = 0.
        outfile.variables['ddt_t'][i,:] = 0.
    for i in np.arange(np.floor(2*(args.n_steps+1)/3.0), args.n_steps+1):
        outfile.variables['ddt_q'][i,:] = -enh*copyfile.variables['ddt_q'][0,:]
        outfile.variables['ddt_t'][i,:] = -enh*copyfile.variables['ddt_t'][0,:]
    for i in np.arange(0,args.n_steps+1):
        outfile.variables['t'][i,:] = copyfile.variables['t'][0,:]
        outfile.variables['u'][i,:] = copyfile.variables['u'][0,:]
        outfile.variables['v'][i,:] = copyfile.variables['v'][0,:]
        outfile.variables['omega'][i,:] = copyfile.variables['omega'][0,:]
        outfile.variables['div'][i,:] = copyfile.variables['div'][0,:]
        outfile.variables['q'][i,:] = copyfile.variables['q'][0,:]
        outfile.variables['ts'][i] = copyfile.variables['ts'][0]
        outfile.variables['aps'][i] = copyfile.variables['aps'][0]
        outfile.variables['lhf'][i] = 0#copyfile.variables['lhf'][0]
    outfile.variables['lon'][:] = copyfile.variables['lon'][:]
    outfile.variables['lat'][:] = copyfile.variables['lat'][:]
    outfile.variables['slm'][:] = copyfile.variables['slm'][:]
    outfile.variables['sn'][:] = copyfile.variables['sn'][:]

    outfile.sync()
    outfile.close()

def add_aerosols(atm, p0, p1, n, m, dn, dm):
    atm.add_layer(p0, p1, n, 'nas')
    atm.add_layer(p0, p1, dn, 'ddt_nas')
    atm.add_layer(p0, p1, 0.2*m, 'ms4as')
    atm.add_layer(p0, p1, 0.2*dm, 'ddt_ms4as')
    atm.add_layer(p0, p1, 0.8*m, 'mduas')
    atm.add_layer(p0, p1, 0.8*dm, 'ddt_mduas')

def add_omega(atm, p0, p1):
    omega = -0.0
    domega = 0

    atm.add_layer(p0, p1, omega, 'omega')
    atm.add_layer(p0, p1, domega, 'ddt_omega')

def write_stdatm(filename):
    print('opening file %s...'%(filename))
    data = Dataset(filename, 'a')
    time = data.variables['time'][:]
    ttot = np.max(time)
    
    # standard atmosphere
    std_t = np.array([15, -56.5, -56.5, -44.5])+tmlt
    std_p = np.array([101325, 22632, 5475, 868])
    r_spec = 287.058
    std_rho = std_p/(r_spec*std_t)
    rh = np.array([0.7, 0.6, 0.5, 0.5])
    
    atm = generic_atmosphere(data, std_p, std_t, rh)

    diss_start = 24*60*60

    dt_mxphase1 = 7.e-5
    dt_mxphase2 = dt_mxphase1*1
    dq_mxphase1 = 15.e-9
    dq_mxphase2 = dq_mxphase1*1
    p_mxphase = [55000, 80000]
    mxphase_start = 0*60*60
    mxphase_end = 12*60*60
    mxphase_dur = mxphase_end-mxphase_start

    dt_ci = 20.e-5
    dq_ci = 15.e-9
    p_ci = [15000, 20000]
    dc = 5.
    ci_start = 3*60*60
    ci_end = 4*60*60
    ci_dur = ci_end-ci_start

    # add aerosols
    n_mx = 75e6
    m_mx = 5e-12
    n_ci = 0.001e6
    m_ci = 0.005e-12
    
    add_aerosols(atm, p_mxphase[0], p_mxphase[1], n_mx, m_mx, 0, 0)
    add_aerosols(atm, p_ci[0], p_ci[1], n_mx, m_mx, 0, 0)

    # add updraft
    add_omega(atm, p_ci[0], p_ci[1])

    # mixed-phase initial relative humidity
    atm.add_rh_layer(p_mxphase[0], p_mxphase[1], 0.8)

    # set rh between clouds to 80% RH_i
    atm.add_rhi_layer(p_ci[1], p_mxphase[0], 0.8)

    # cirrus initial relative humidity
    atm.add_rh_layer(p_ci[0], p_ci[1], 0.65)

    # mixed-phase temperature tendency
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], mxphase_start, mxphase_end, -dt_mxphase1, -dt_mxphase2, 'ddt_t')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], mxphase_end, diss_start, 0, 0, 'ddt_t')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], diss_start, diss_start+mxphase_dur, dt_mxphase1, dt_mxphase2, 'ddt_t')

    # mixed-phase humidity tendency
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], mxphase_start, mxphase_end, dq_mxphase1, dq_mxphase2, 'ddt_q')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], mxphase_end, diss_start, 0, 0, 'ddt_q')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], diss_start, diss_start+mxphase_dur, -dq_mxphase1, -dq_mxphase2, 'ddt_q')

    # cirrus temperature tendency
    atm.add_layer_segment(p_ci[0], p_ci[1], ci_start, ci_end, -dt_ci, -dt_ci, 'ddt_t')
    atm.add_layer_segment(p_ci[0], p_ci[1], ci_end, diss_start, 0, 0, 'ddt_t')
    atm.add_layer_segment(p_ci[0], p_ci[1], diss_start, diss_start+ci_dur, dt_ci, dt_ci, 'ddt_t')

    # cirrus humidity tendency
    atm.add_layer_segment(p_ci[0], p_ci[1], ci_start, ci_end, dq_ci, dq_ci, 'ddt_q')
    atm.add_layer_segment(p_ci[0], p_ci[1], ci_end, diss_start, 0, 0, 'ddt_q')
    atm.add_layer_segment(p_ci[0], p_ci[1], diss_start, diss_start+ci_dur, -dq_ci, -dq_ci, 'ddt_q')

    data.sync()
    data.close()

def write_deep(filename):
    print('opening file %s...'%(filename))
    data = Dataset(filename, 'a')
    time = data.variables['time'][:]
    ttot = np.max(time)

    # standard atmosphere
    std_t = np.array([15, -56.5, -56.5, -44.5])+tmlt
    std_p = np.array([101325, 22632, 5475, 868])
    r_spec = 287.058
    std_rho = std_p/(r_spec*std_t)
    rh = np.array([0.6, 0.3, 0.3, 0.3])
    
    # # standard atmosphere
    # std_t = np.array([0, -40, -40, -34.5])+tmlt
    # std_p = np.array([101325, 22632, 5475, 868])
    # r_spec = 287.058
    # std_rho = std_p/(r_spec*std_t)
    # rh = np.array([0.6, 0.3, 0.3, 0.3])
    
    atm = generic_atmosphere(data, std_p, std_t, rh)

    dt_mxphase = 7.e-5
    dq_mxphase1 = 2e-9
    dq_mxphase2 = 6e-9
    p_mxphase = [15000, 80000]
    dm = 3.

    # mixed-phase initial relative humidity
    atm.add_rh_layer(p_mxphase[0], p_mxphase[1], 0.8)

    # mixed-phase temperature tendency
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], 0, ttot/dm, -dt_mxphase, -dt_mxphase, 'ddt_t')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], ttot/dm, (dm-1)*ttot/dm, 0, 0, 'ddt_t')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], (dm-1)*ttot/dm, ttot, dt_mxphase, dt_mxphase, 'ddt_t')

    # mixed-phase humidity tendency
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], 0, ttot/dm, dq_mxphase1, dq_mxphase2, 'ddt_q')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], ttot/dm, (dm-1)*ttot/dm, 0, 0, 'ddt_q')
    atm.add_layer_segment(p_mxphase[0], p_mxphase[1], (dm-1)*ttot/dm, ttot, -dq_mxphase1, -dq_mxphase2, 'ddt_q')

    data.sync()
    data.close()

def write_warm(filename):
    print('opening file %s...'%(filename))
    data = Dataset(filename, 'a')
    time = data.variables['time'][:]
    ttot = np.max(time)

    temperature = np.array([20, 20])+tmlt
    pressure = np.array([101325, 800])
    relhum = np.array([1, 1])

    p_cloud = [40000, 90000]
    dq_cloud = 5e-7
        
    atm = generic_atmosphere(data, pressure, temperature, relhum)

    atm.add_layer_segment(p_cloud[0], p_cloud[1], 0, ttot, dq_cloud, dq_cloud, 'ddt_q')

    data.sync()
    data.close()

def write_numerics(filename):
    print('opening file %s...'%(filename))
    data = Dataset(filename, 'a')
    time = data.variables['time'][:]
    ttot = np.max(time)
    
    # standard atmosphere
    # std_t = np.array([15, -56.5, -56.5, -44.5])+tmlt
    std_t = np.array([15, -56.5, -56.5, -44.5])+tmlt-15
    std_p = np.array([101325, 22632, 5475, 868])
    r_spec = 287.058
    std_rho = std_p/(r_spec*std_t)

    rh = np.array([0.1, 0.1, 0.1, 0.1])

    atm = generic_atmosphere(data, std_p, std_t, rh)

    p0 = 40000
    p1 = 45000
    dqi = 5e-6
    dni = 2*dqi*1e8
    dqirim = dqi
    dbirim = dqirim/900.0

    # injection relative humidity
    atm.add_rh_layer(p0, p1, 1.0)
    atm.add_rh_layer(868, 101325, 0.5)

    atm.add_layer_segment(p0, p1, 0, ttot+1000, dqi, dqi, 'ddt_qi')
    atm.add_layer_segment(p0, p1, 0, ttot+1000, dqirim, dqirim, 'ddt_qirim')
    atm.add_layer_segment(p0, p1, 0, ttot+1000, dni, dni, 'ddt_ni')
    atm.add_layer_segment(p0, p1, 0, ttot+1000, dbirim, dbirim, 'ddt_birim')

    data.sync()
    data.close()

def write_infinite_cirrus(filename):
    print('opening file %s...'%(filename))
    data = Dataset(filename, 'a')
    time = data.variables['time'][:]
    ttot = np.max(time)

    # standard atmosphere
    std_t = np.array([15, -56.5, -56.5, -44.5])+tmlt
    std_p = np.array([101325, 22632, 5475, 868])
    r_spec = 287.058
    std_rho = std_p/(r_spec*std_t)
    rh = np.array([0.6, 0.3, 0.3, 0.3])
    atm = generic_atmosphere(data, std_p, std_t, rh)

    # add aerosols
    n_aero = 75e6
    m_aero = 1e-9
    # n_aero = 0
    # m_aero = 0

    dt_ci = 20.e-5
    dq_ci = 1.e-9
    p = [15000, 20000]

    add_aerosols(atm, p[0], p[1], n_aero, m_aero, 0, 0)
    atm.add_rhi_layer(p[0], p[1], 1.3)
    atm.add_layer(p[0], p[1], dt_ci, 'ddt_t')
    atm.add_layer(p[0], p[1], dq_ci, 'ddt_q')

    data.sync()
    data.close()

def main():
    parser = get_parser()
    args = parser.parse_args()

    create_file(args.filename, args.n_steps, args.delta_time)

    if(args.copyname):
        copy_from_file(args.filename, args.copyname)
    if(args.std_atm):
        write_stdatm(args.filename)
    if(args.deep):
        write_deep(args.filename)
    if(args.warm):
        write_warm(args.filename)
    if(args.numerics):
        write_numerics(args.filename)
    if(args.infinite_cirrus):
        write_infinite_cirrus(args.filename)

if __name__ == '__main__':
    main()

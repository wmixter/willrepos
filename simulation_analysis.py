#!/usr/bin/env python3
"""
Usage:
    simulation_analysis  <filename> <num_of_neutrons> [options]

Options:
   -h  --help           This information
   --carbon             Filters for only carbon atoms based on gamma energy
   --data               Save the result data in hdf5 file
   --density           Creates several contour plots for density
       
   --depth              Creates error plots versus depth into soil
   --energyhist         Creates a histogram  for gamma energy
   --ergerror           Creates a plot of error versus gamma energy
   --g-alphapos ERROR   [default: 0] unit: cm FWHM 
   --g-alphatime ERROR  [default: 0] unit: ns FWHM 
   --g-gammatime ERROR  [default: 0] unit: ns WHM 
   --gsweep-alphapos RANGE    Sweeps gaussian error for alpha position
   --gsweep-alphatime RANGE     Sweeps gaussian error for alpha time
   --gsweep-gammaenergy RANGE   Sweeps gaussian error for gamma energy
   --gsweep-gammatime RANGE     Sweeps gaussian error for gamma time
   --plotall            Creates a 3D plot of all of the data. WARNING: slow for large data sets
   --rate RATE          Neutron rate [default: 2e8] unit: neutrons/second
   --rateacc RANGE      Accuracy versus rate. Format: START,FINISH where
                           START/FINISH are exponents of the range values.
                           (ex. range from 1 to 1e9; enter 0,9)
   --ratesweep RANGE    Gamma/alpha pair versus rate. Format: START,FINISH where
                        START/FINISH are exponents of the range values.
                        (ex. range from 1 to 1e9; enter 0,9)
   --seed SEED          Seed
   --single INDEX
   --time TIME          Unit: minutes. simulate only a certain amount of time

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.constants as spc
from docopt import docopt
import timeit


from calculate import calculate


commands = docopt(__doc__, version='1.0')
print(commands)
filename = commands['<filename>']
rate = float(commands['--rate']) #neutrons/second
num_parts = float(commands['<num_of_neutrons>']) #total neutrons in simulation
carbon = commands['--carbon'] #true/false
ratesweep = commands['--ratesweep'] #true/false
st1 = timeit.default_timer()


with h5py.File(filename, 'r') as hdf:
    #ls = list(hdf.keys())
    #print('list of dataset in this file: \n', ls)

    #Create numpy array for each dataset
    #comment out any unwanted datasets
   # all_data = np.array(hdf.get('all_data'))
    alpha_y = np.array(hdf.get('alpha_y[cm]'), dtype=np.float64)
    alpha_x = np.array(hdf.get('alpha_x[cm]'), dtype=np.float64)
    alpha_t = np.array(hdf.get('alpha_t[s]'), dtype=np.float64)
    atom_type = np.array(hdf.get('atom_type'))
    atom_x = np.array(hdf.get('atom_x[cm]'), dtype=np.float64)
    atom_y = np.array(hdf.get('atom_y[cm]'), dtype=np.float64)
    atom_z = np.array(hdf.get('atom_z[cm]'), dtype=np.float64)
    gamma_t = np.array(hdf.get('gamma_t[s]'), dtype=np.float64)
    gamma_e = np.array(hdf.get('gamma_e[MeV]'), dtype=np.float64)
    exact_gamma_x = np.array(hdf.get('gamma_x[cm]'), dtype=np.float64)
    exact_gamma_y = np.array(hdf.get('gamma_y[cm]'), dtype=np.float64)
    exact_gamma_z = np.array(hdf.get('gamma_z[cm]'), dtype=np.float64)


if commands['--seed']:
    seed = int(commands['--seed'])
    print('Seed is:', seed)
else:
    seed = np.random.randint(100000, size=1)[0]
    print('Seed is:', seed)
RN_gauss = np.random.RandomState(seed)

#Set the point location of the gamma detector
gx = 26.65
gy = 0
gz = 50
gamma_x = np.ones_like(exact_gamma_x)*gx #no error
gamma_y = np.zeros_like(exact_gamma_y)*gy #no error
gamma_z = np.ones_like(exact_gamma_z)*gz #no error

alpha_z = -6*np.ones_like(alpha_y) #location of detector

#### Velocity Calculations

# alpha velocity calculation
MeV_alpha = 3.5 #MeV
energy_alpha = MeV_alpha*spc.physical_constants['electron volt-joule relationship'][0]*(1E6) #Joules
mass_alpha = spc.physical_constants['alpha particle mass'][0]#kg
v_alpha = np.sqrt((2*energy_alpha)/mass_alpha)*100 #cm/s
r_alpha = np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2)
calc_t_alpha = r_alpha/v_alpha

# gamma velocity
c = (spc.c)*100 #cm/s

# neutron velocity 
MeV_neut = 14.0 #MeV
energy_neut = MeV_neut*spc.physical_constants['electron volt-joule relationship'][0]*(1E6) #Joules
mass_neut = spc.m_n #kg
v_neut = np.sqrt((2*energy_neut)/mass_neut)*100 #cm/s


##### Location of neutron source
neutron_x = 0
neutron_y = 0
neutron_z = 0

##### Unit vector components for neutron velocity
vmag = np.sqrt((neutron_x-alpha_x)**2 + (neutron_y-alpha_y)**2 + (neutron_z-alpha_z)**2)
vx = (neutron_x-alpha_x)/vmag
vy = (neutron_y-alpha_y)/vmag
vz = (neutron_z-alpha_z)/vmag

#### Calculate timing window**
#Calculate shortest time
rn_min = np.array([0, 0, 60], dtype='float64') #neutron vector
ra_min = np.array([0, 0, -6], dtype='float64') #alpha vector
rg = np.array([gx, gy, gz], dtype='float64') #gamma vector
dg_min = rg - rn_min #distance traveled by gamma
tn_min = np.linalg.norm(rn_min)/v_neut #time traveled by neutron
ta_min = np.linalg.norm(ra_min)/v_alpha #time traveled by alpha
tg_min = np.linalg.norm(dg_min)/c #time traveled by gamma
t_min = tn_min + tg_min - ta_min #Minimum time
#Calculate longest time
rn_max = np.array([-25, 25, 90]) #neutron vector
theta = np.arctan((25*(np.sqrt(2)))/90) #Aimuthal angle of neutron vector
mag_ra_max = 6/np.cos(theta) #Max distance of alpha
dg_max = rg - rn_max #distance traveled by gamma
tn_max = np.linalg.norm(rn_max)/v_neut #time traveled by neutron
ta_max = mag_ra_max/v_alpha #time traveled by alpha
tg_max = np.linalg.norm(dg_max)/c #time traveled by gamma
t_max = tn_max + tg_max - ta_max #Maximum time


#### Gaussian Error
gauss_alpha_pos = float(commands['--g-alphapos'])/(2*np.sqrt(2*np.log(2)))/np.sqrt(3)
gauss_alpha_time = float(commands['--g-alphatime'])/(2*np.sqrt(2*np.log(2)))*1e-9
gauss_gamma_time = float(commands['--g-gammatime'])/(2*np.sqrt(2*np.log(2)))*1e-9
gauss_gamma_erg = 0

gauss = np.array([gauss_alpha_pos, gauss_alpha_time, gauss_gamma_time,
                  gauss_gamma_erg], dtype='float64')

alpha_x = alpha_x + RN_gauss.normal(scale=gauss_alpha_pos, size=len(alpha_x))
alpha_y = alpha_y + RN_gauss.normal(scale=gauss_alpha_pos, size=len(alpha_y))
alpha_z = alpha_z + RN_gauss.normal(scale=gauss_alpha_pos, size=len(alpha_z))
alpha_t = alpha_t + RN_gauss.normal(scale=gauss_alpha_time, size=len(alpha_t))
gamma_t = gamma_t + RN_gauss.normal(scale=gauss_gamma_time, size=len(gamma_t))


#### Create input variables
alpha = np.column_stack((alpha_x, alpha_y, alpha_z, alpha_t, calc_t_alpha,
                         alpha_x, alpha_y, alpha_z, alpha_t))
neutron = np.column_stack((vx, vy, vz))
gamma = np.column_stack((gamma_x, gamma_y, gamma_z, gamma_t, gamma_e, exact_gamma_x,
                         exact_gamma_y, exact_gamma_z, gamma_t, gamma_e))
atom = np.column_stack((atom_x, atom_y, atom_z, atom_type))

#### Vary the time of the simulation
if commands['--time']:
    duration = int(commands['--time'])*60
    N = rate*duration
    cutoff = int((N/num_parts)*len(alpha))
    alpha = alpha[0:cutoff,:]
    gamma = gamma[0:cutoff,:]
    neutron = neutron[0:cutoff, :]
    atom = atom[0:cutoff, :]
    num_parts = N








#### Create Density maps
if commands['--density']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    voxel = 1
    xmin = ymin = -60
    xmax = ymax = 60
    zmin = 50
    zmax = 120
    X = np.arange(xmin,xmax + voxel, voxel)
    Y = np.arange(ymin,ymax + voxel, voxel)
    Z = np.arange(zmin,zmax + voxel, voxel)


    H, edges = np.histogramdd(calc.calculated_atom, bins = (X, Y, Z))
    
    plt.figure(num=1, figsize=(15, 8))
    plt.subplot(111)
    c = plt.imshow(np.rot90(H[:,58,:] + H[:,59,:] + H[:,60,:] + H[:,61,:] + H[:,62,:], 3), extent = [-60, 60, 60, -10])
    plt.title('{}<y<{}   {}_min-{}_sec-'.format(edges[1][58], edges[1][63], calc.secs//60%60, int(calc.secs % 60)))
    plt.colorbar(c)
    
    
    plt.figure(num=2, figsize=(15, 8))
    bins = 10000
    plt.hist(calc.errordistance, bins, histtype='step', normed=True)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.title('Without gaussian error')
    plt.xlabel('Error distance [cm]')
    plt.ylabel('Normalized Counts')
    plt.xlim([0, 10])

    #plt.show()


#### Plot error as a function of depth
if commands['--depth']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    print('Error less than 5cm:', calc.percent, '%')
    fig8 = plt.figure(8)
    plt.scatter(calc.atom[:, 2], calc.errordistance, marker='.', c='blue', label='actual distance')
    #plt.scatter(calc.calculated_atom[:,2],calc.errordistance, marker='.', c='red', label='calculated distance')
    plt.title('Error distance versus atom depth')
    plt.xlabel('Actual atom depth [cm]')
    plt.ylabel('Error distance [cm]')
    #plt.legend(loc='upper left')
    plt.ylim([0, 100])
    plt.xlim([50, 130])
    plt.savefig('error_vs_depth'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()

    z = 60
    Y = []
    Z = np.arange(60, 85, 5)
    for z in Z:
        bad = 0
        count = 0
        counttotal = np.where(np.logical_and(calc.atom[:, 2] >= z, calc.atom[:, 2] <= z+5))
        #print(counttotal)
        for e in calc.errordistance[counttotal]:
            count += 1
            if e > 5:
                bad += 1
        if count == 0:
            Y.append(0)
        else:
            Y.append((bad/count)*100)


    fig9 = plt.figure(9)
    plt.plot(Z, Y, marker='.', c='blue', label='actual distance')
    plt.title('Error distance versus atom depth')
    plt.xlabel('Actual atom depth [cm]')
    plt.ylabel('Percent with error >5cm [%]')
    #plt.legend(loc='upper left')
    #plt.ylim([0,100])
    plt.xlim([50, 90])
    plt.savefig('histogram_error_depth-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()

#### Plot energy spectrum histogram
if commands['--energyhist']:
    fig4 = plt.figure(4)
    bins = 100
    #energy = [energy[a] for a in range(len(energy)) if energy[a]<10]
    plt.hist(gamma[:, 4], bins, histtype='step')
    plt.xlabel('energy')
    #plt.ylabel('counts')
    plt.title('Energy spectrum at gamma detector from {}'.format(filename))
    #plt.set_xlim([0,10])
    plt.savefig('energyhist-'+filename+'.png')
    #plt.show()

#### Plot error distribution as a function of energy
if commands['--ergerror']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    erg = np.linspace(0, 8, 17)
    counttotal = np.zeros_like(erg)
    countbad = np.zeros_like(erg)
    for i in range(len(calc.gamma)):
        counttotal[np.searchsorted(erg, calc.gamma[i, 4])-1] += 1
        if calc.errordistance[i] > 5:
            countbad[np.searchsorted(erg, calc.gamma[i, 4])-1] += 1
    count = countbad/counttotal*100
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    print('Percent of nuclei kept:', calc.keptnuclei)
    fig6 = plt.figure(6)
    plt.bar(erg, count, ec='black', width=.5, align='edge')
    #plt.scatter(gauss_alpha_time,sweeppercent, marker='.')
    plt.title('Error versus energy')
    plt.xlabel('Gamma energy [MeV]')
    plt.ylabel('% with error >5cm')
    #plt.savefig('gauss-sweep-alpha_time-'+filename+'seed-{}'.format(seed)+'.png')
    plt.show()

#### Sweep gaussian error in alpha position
if commands['--gsweep-alphapos']:
    gauss_alpha_pos = commands['--gsweep-alphapos']
    gauss_alpha_pos = gauss_alpha_pos.split(',')
    start = float(gauss_alpha_pos[0])
    stop = float(gauss_alpha_pos[1])
    print('Alpha position error from {} to {}'.format(start, stop))
    num = 50
    gauss_alpha_pos = np.linspace(start, stop, num=num)
    gauss_alpha_pos_sigma = gauss_alpha_pos/(2*np.sqrt(2*np.log(2)))/np.sqrt(3)

    sweepapos = []
    count = 0
    for g in gauss_alpha_pos_sigma:
        alphag = np.copy(alpha)

        print('Percent Complete', int((count/num)*100), '%')
        alphag[:, 0] = alpha[:, 0] + RN_gauss.normal(scale=g, size=len(alpha))
        alphag[:, 1] = alpha[:, 1] + RN_gauss.normal(scale=g, size=len(alpha))
        alphag[:, 2] = alpha[:, 2] + RN_gauss.normal(scale=g, size=len(alpha))
        r_alpha1 = np.sqrt(alphag[:, 0]**2 + alphag[:, 1]**2 + alphag[:, 2]**2)
        calc_t_alpha1 = r_alpha1/v_alpha
        vmag1 = np.sqrt(alphag[:, 0]**2 + alphag[:, 1]**2 + alphag[:, 2]**2)
        vx1 = -alphag[:, 0]/vmag1
        vy1 = -alpha[:, 1]/vmag1
        vz1 = -alpha[:, 2]/vmag1
        neutron1 = np.column_stack((vx1, vy1, vz1))
        calc = calculate(alphag, gamma, c, neutron1, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        sweepapos.append(calc.percent)
        count += 1
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    print('Percent of nuclei kept:', calc.keptnuclei)
    if not commands['--gsweep-alphatime']:
        fig5 = plt.figure(5)
        plt.plot(gauss_alpha_pos, sweepapos)
        plt.scatter(gauss_alpha_pos, sweepapos, marker='.')
        plt.title('Accuracy vs Alpha Position Error')
        plt.xlabel(r'$\sigma$ of gaussian error in alpha position [cm]')
        plt.ylabel('Percent of atoms within 5cm error [%]')
        plt.ylim([0, 100])
        plt.savefig('gauss-sweep-alpha_pos-'+filename+'seed-{}'.format(seed)+'.png')
        #plt.show()

#### Sweep gaussian error in alpha time
if commands['--gsweep-alphatime']:
    gauss_alpha_time = commands['--gsweep-alphatime']
    gauss_alpha_time = gauss_alpha_time.split(',')
    start = float(gauss_alpha_time[0])
    stop = float(gauss_alpha_time[1])
    print('Alpha time error from {} ns to {} ns'.format(start, stop))
    num = 50
    gauss_alpha_time = np.linspace(start, stop, num=num)
    gauss_alpha_time_sigma = gauss_alpha_time/(2*np.sqrt(2*np.log(2)))*1e-9
    sweepatime = []
    alphag = np.copy(alpha)
    count = 0
    for g in gauss_alpha_time_sigma:
        print('Percent Complete', int((count/num)*100), '%')
        alphag[:, 3] = alpha[:, 3] + RN_gauss.normal(scale=g, size=len(alpha))
        calc = calculate(alphag, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        sweepatime.append(calc.percent)
        count += 1
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    print('Percent of nuclei kept:', calc.keptnuclei)
    if not commands['--gsweep-alphapos']:
        fig6 = plt.figure(6)
        plt.plot(gauss_alpha_time, sweepatime)
        plt.scatter(gauss_alpha_time, sweepatime, marker='.')
        plt.title('Accuracy vs Alpha Time Error')
        plt.xlabel(r'$\sigma$ of gaussian error in alpha time [ns]')
        plt.ylabel('Percent of atoms within 5cm error [%]')
        plt.ylim([0, 100])
        plt.savefig('gauss-sweep-alpha_time-'+filename+'seed-{}'.format(seed)+'.png')
        #plt.show()

#### Sweep gaussian error in gamma time
if commands['--gsweep-gammatime']:
    gauss_gamma_time = commands['--gsweep-gammatime']
    gauss_gamma_time = gauss_gamma_time.split(',')
    start = float(gauss_gamma_time[0])
    stop = float(gauss_gamma_time[1])
    print('Gamma time error from {} ns to {} ns'.format(start, stop))
    num = 50
    gauss_gamma_time = np.linspace(start, stop, num=num)
    gauss_gamma_time_sigma = gauss_gamma_time/(2*np.sqrt(2*np.log(2)))*1e-9
    sweepgtime = []
    gammag = np.copy(gamma)
    count = 0
    for g in gauss_gamma_time_sigma:
        print('Percent Complete', int((count/num)*100), '%')
        gammag[:, 3] = gamma[:, 3] + RN_gauss.normal(scale=g, size=len(gamma))
        calc = calculate(alpha, gammag, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        sweepgtime.append(calc.percent)
        count += 1
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    print('Percent of nuclei kept:', calc.keptnuclei)
    if not commands['--gsweep-alphapos']:
        fig7 = plt.figure(7)
        plt.plot(gauss_gamma_time, sweepgtime)
        plt.scatter(gauss_gamma_time, sweepgtime, marker='.')
        plt.title('Accuracy vs Gamma Time Error')
        plt.xlabel(r'$\sigma$ of gaussian error in gamma time [ns]')
        plt.ylabel('Percent of atoms within 5cm error [%]')
        plt.ylim([0, 100])
        plt.savefig('gauss-sweep-gamma_time-'+filename+'seed-{}'.format(seed)+'.png')
        #plt.show()
    
#### Plot gaussian sweep for all three on one graph
if commands['--gsweep-alphapos'] and commands['--gsweep-alphatime'] and commands['--gsweep-gammatime']:
    fig, ax1 = plt.subplots(figsize=(15, 8), facecolor='w', edgecolor='k')

    color = 'tab:red'
    ax1.set_xlabel('FWHM of gaussian error in position [mm]', size=25, color=color)
    ax1.plot(gauss_alpha_pos*10, sweepapos, linestyle ='-.', marker='o', color=color, label='alpha position error')
    #ax1.plot(np.linspace(start, stop, num=num), sweepnpos, linestyle ='--', marker='>', color=color, label='neutron position error')
    ax1.tick_params(axis='x', labelcolor=color, labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel('Atoms within 5cm error [%]', size=25)
    ax2 = ax1.twiny()
    color = 'tab:blue'
    ax2.set_xlabel('FWHM of gaussian error in time [ns]', size=25, labelpad=10)
    #ax2.set_ylabel('Percent of atoms within 5cm error [%]', size=20)
    ax2.plot(gauss_gamma_time, sweepgtime, linestyle=':', marker='*', color=color, label='gamma time error')
    ax2.plot(gauss_alpha_time, sweepatime, linestyle='-', marker='^', color= color, label='alpha time error')
    ax2.tick_params(axis='x', labelcolor=color, labelsize=20)
    ax2.set_xlabel('FWHM of gaussian error in time [ns]', size=25, color=color)
    #fig.tight_layout()
    font = {'size'   : 20}
    plt.ylim([0, 100])
    plt.rc('font', **font)
    fig.legend(bbox_to_anchor=(.15, .15), loc=3, borderaxespad=0.)
    plt.savefig('gauss-sweep-all-'+filename+'seed-{}'.format(seed)+'.pdf')
    plt.show()

#### Sweep gaussian error in gamma energy
if commands['--gsweep-gammaenergy']:
    carbon = True
    gauss_gamma_erg = commands['--gsweep-gammaenergy']
    gauss_gamma_erg = gauss_gamma_erg.split(',')
    start = float(gauss_gamma_erg[0])
    stop = float(gauss_gamma_erg[1])
    print('Gamma energy error from {}% to {}%'.format(start, stop))
    num = 50
    gauss_gamma_energy = np.linspace(start/100*4.439, stop/100*4.439, num=num)
    estcarbon = []
    gammag = np.copy(gamma)
    pcarb = []
    pox = []
    pother = []
    psi = []
    count = 0
    for g in gauss_gamma_energy:
        print('Percent Complete', int((count/num)*100), '%')
        gammag[:, 4] = gamma[:, 4] + RN_gauss.normal(scale=g, size=len(gamma))
        gauss[3] = g
        calc = calculate(alpha, gammag, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        estcarbon.append((calc.number_carbon/calc.acceptedsamplesize)*100)
        pcarb.append((calc.actualcarbon/calc.number_carbon)*100)
        pox.append((calc.actualoxygen/calc.number_carbon)*100)
        pother.append((calc.others/calc.number_carbon)*100)
        psi.append((calc.actualsilicon/calc.number_carbon)*100)
        count += 1
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    print('Percent of nuclei kept:', calc.keptnuclei)
    fig9 = plt.figure(9)
    plt.plot(gauss_gamma_energy/4.439*100, estcarbon)
    plt.scatter(gauss_gamma_energy/4.439*100, estcarbon, marker='.')
    plt.title('% gammas chosen as carbon versus gamma energy error')
    plt.xlabel('Stdv. of gaussian error in gamma energy [%]')
    plt.ylabel('Percent gammas chosen as carbon [%]')
    plt.ylim([0, 20])
    plt.savefig('gauss-sweep-gamma_energy_kept-'+filename+'seed-{}'.format(seed)+'.png')

    fig10 = plt.figure(10)
    plt.plot(gauss_gamma_energy/4.439*100, pcarb, c='red', label='carbon')
    plt.scatter(gauss_gamma_energy/4.439*100, pcarb, marker='.', c='red')
    plt.plot(gauss_gamma_energy/4.439*100, pox, c='blue', label='oxygen')
    plt.scatter(gauss_gamma_energy/4.439*100, pox, marker='.', c='blue')
    plt.plot(gauss_gamma_energy/4.439*100, pother, c='green', label='other')
    plt.scatter(gauss_gamma_energy/4.439*100, pother, marker='.', c='green')
    plt.plot(gauss_gamma_energy/4.439*100, psi, c='cyan', label='silicon')
    plt.scatter(gauss_gamma_energy/4.439*100, psi, marker='.', c='cyan')
    plt.title('Atoms Chosen to be Carbon vs Gamma Energy Error')
    plt.xlabel(r'$\sigma$ of gaussian error in gamma energy [%]')
    plt.ylabel('Atoms within carbon energy window [%]')
    plt.ylim([0, 100])
    plt.legend(loc='upper left')
    plt.savefig('gauss-sweep-gamma_energy_atom-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()

#### Plot all events
if commands['--plotall']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    print('Sample size:', calc.samplesize)
    print('Rate:', rate)
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    if carbon is not False:
        print('Carbon energy window: {} MeV to {} MeV'.format(calc.carbonlowerlimit, calc.carbonupperlimit))
        print('Percent of carbon atoms:', (calc.number_carbon/calc.notcarbon)*100, '%')
    print('Percent of nuclei kept:', calc.keptnuclei)
    print('Error less than 5cm:', calc.percent)

    fig3 = plt.figure(num=3, figsize=(15, 8), facecolor='w', edgecolor='k')

    axis3 = fig3.gca(projection='3d')
    axis3.set_proj_type('ortho') #make z-axis orthogonal

    axis3.scatter(calc.atom[:, 0], calc.atom[:, 1], calc.atom[:, 2], c='blue', s=10, label='exact nuclei')
    axis3.scatter(calc.alpha[:, 5], calc.alpha[:, 6], calc.alpha[:, 7], c='cyan', s=10, label='alpha')
    axis3.scatter(calc.gamma[:, 5], calc.gamma[:, 6], calc.gamma[:, 7], c='yellow', s=10, label='gamma detected')
    axis3.scatter(calc.gamma[0, 0],calc.gamma[0, 1], calc.gamma[0, 2], c='green', s=50, label='Point detector location')
    
    nx = np.zeros_like(calc.alpha[:, 0])
    ny = np.zeros_like(calc.alpha[:, 0])
    nz = np.zeros_like(calc.alpha[:, 0])
    axis3.scatter(nx, ny, nz, c='orange', s = 10, label='origin')

    axis3.scatter(calc.calculated_atom[:, 0], calc.calculated_atom[:, 1], calc.calculated_atom[:, 2], c='red', s = 10, label='calculated nuclei')
    
    plt.title('Overall Plot'.format(filename))
    axis3.set_xlim([-40, 40])
    axis3.set_ylim([-40, 40])
    axis3.set_zlim([-10, 120])
    
    plt.legend(loc='upper left')
    
    axis3.set_xlabel('x (cm)')
    axis3.set_ylabel('y')
    axis3.set_zlabel('z (cm)')
    axis3.xaxis.labelpad = 10 #move xlabel downward
    
    #Turn off y-axis
    axis3.w_yaxis.line.set_lw(0.)
    axis3.set_yticks([])
    #axis view
    axis3.view_init(180, 90)
    plt.draw()
    
    #plt.savefig('plotall-'+filename+'.png')
    plt.show()
    
    
#### Plot accuracy as a function of rate
if commands['--rateacc']:
    rate = commands['--rateacc']
    rate = rate.split(',')
    start = float(rate[0])
    stop = float(rate[1])
    print('Rate range from 1e{} to 1e{} neutrons/second without extra alphas'.format(start,stop))
    num= 50 
    rate = np.logspace(start,stop,num=num)
    acc = []
    sec = []
    count = 0
    for r in rate:
        print('Percent Complete', int((count/num)*100), '%')
        calc = calculate(alpha, gamma, c, neutron, v_neut, atom, r, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        acc.append(calc.percent)
        sec.append(calc.secs)
        count += 1
    fig4 = plt.figure(4)
    plt.semilogx(sec, acc)
    plt.scatter(sec, acc, marker='.')
    plt.title('Accuracy vs Simulation Duration. ')
    plt.xlabel('(Total Number of Samples)/(Rate) [s]')
    plt.ylabel('Percent of atoms within 5cm error [%]')
    plt.savefig('accuracy_vs_rate-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()

#### Sweep neutron rate and plot versus successful alpha/gamma pairs
if commands['--ratesweep']:
    ratesweep = True
    rate1 = commands['--ratesweep']
    rate1 = rate1.split(',')
    start = float(rate1[0])
    stop = float(rate1[1])
    print('Rate range from 1e{} to 1e{} neutrons/second'.format(start, stop))
    num = 50
    rate1 = np.logspace(start,stop,num=num)
    keptnuclei = []
    count = 0
    for r in rate1:
        print('rate', r)
        print('Percent Complete',  int((count/num)*100), '%')
        calc = calculate(alpha, gamma, c, neutron, v_neut, atom, r, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
        keptnuclei.append(calc.keptnuclei)
        count += 1
    fig4= plt.figure(num=4, figsize=(15, 8), facecolor='w', edgecolor='k')
    plt.semilogx(rate1, keptnuclei)
    plt.scatter(rate1, keptnuclei, marker='.')
    plt.axvline(x=2e8, color='red') #Add vertical line
    #plt.title('Gamma/Alpha matches versus rate', size=25)
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.xlabel('rate [neutrons/second]', size=30)
    plt.ylabel('gamma/alpha matches [%]', size=30)
    plt.savefig('ratesweep-'+filename+'seed-{}'.format(seed)+'.pdf')
    #plt.show()


#### plot all events for a single history
if commands['--single']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    print('Error less than 5cm:', calc.percent, '%')
    #index for reporting
    i = int(commands['--single'])
    print('index is', i)
    print('exact atom location is:', calc.atom[i,0], calc.atom[i,1], calc.atom[i,2], 'cm')
    print('calculated atom location is:', calc.calculated_atom[i,0], calc.calculated_atom[i,1], calc.calculated_atom[i,2], 'cm')
    print('error distance is', calc.errordistance[i], 'cm')
    plt.close()
    fig2 = plt.figure(2)
    axis = fig2.gca(projection='3d')
    axis.set_proj_type('ortho') #make z-axis orthogonal

    axis.scatter(calc.atom[i, 0], calc.atom[i, 1], calc.atom[i, 2], c='blue', s=10, label='nuclei')
    axis.scatter(calc.alpha[i, 0], calc.alpha[i, 1], calc.alpha[i, 2], c='cyan', s=10, label='alpha')
    axis.scatter(calc.gamma[i, 0], calc.gamma[i, 1], calc.gamma[i, 2], c='yellow', s=10, label='gamma detected')

    nx = 0
    ny = 0
    nz = 0
    axis.scatter(nx, ny, nz, c='orange', s=10, label='origin')



    axis.plot([calc.alpha[i, 0]], nx, [calc.alpha[i, 1]], ny, [calc.alpha[i, 2]], nz, label='origin')

    axis.plot([nx, calc.calculated_atom[i, 0]], [ny, calc.calculated_atom[i, 1]], [nz, calc.calculated_atom[i, 2]], color='red', label='calculated')
    axis.scatter(calc.calculated_atom[i, 0], calc.calculated_atom[i, 1], calc.calculated_atom[i, 2], c='red', s=10, label='calculated')

    plt.title('Gammas from Carbon graph from {}'.format(filename))
    axis.set_xlim([-40, 40])
    axis.set_ylim([-40, 40])
    axis.set_zlim([-10, 120])

    plt.legend(loc='upper left')

    axis.set_xlabel('x (cm)')
    axis.set_ylabel('y')
    axis.set_zlabel('z (cm)')
    axis.xaxis.labelpad = 10 #move xlabel downward

    #axis view
    #axis.view_init(0, -90)
    plt.draw()
    plt.show()
    
    
    #### Save data
if commands['--data']:
    calc = calculate(alpha, gamma, c, neutron, v_neut, atom, rate, num_parts, seed, gauss, t_min, t_max, carbon, ratesweep)
    print('Sample size:', calc.samplesize)
    print('Rate:',rate)
    print('Duration=', calc.secs//60//60, 'hours', calc.secs//60%60, 'minutes', calc.secs % 60, 'seconds')
    print('Gamma/alpha time window: {} sec to {} sec'.format(calc.mindiff, calc.maxdiff))
    print('More than one gamma per alpha:', calc.morethanone_gamma)
    print('No gamma per alpha:', calc.nogamma)
    print('More than one alpha per gamma:', calc.morethanone_alpha)
    print('No alpha per gamma:', calc.noalpha)
    if carbon is not False:
        print('Carbon energy window: {} MeV to {} MeV'.format(calc.carbonlowerlimit, calc.carbonupperlimit))
        print('Percent of estimated carbon atoms:',(calc.number_carbon/calc.acceptedsamplesize)*100, '%')
        print('Percent of actual carbon atoms:', (calc.actualcarbon/calc.number_carbon)*100, '%')
        print('Percent of actual oxygen atoms:', (calc.actualoxygen/calc.number_carbon)*100, '%')
        print('Percent of actual silicon atoms:', (calc.actualsilicon/calc.number_carbon)*100, '%')
        print('Percent of other atoms:', (calc.others/calc.number_carbon)*100, '%')
    print('Percent of nuclei kept:', calc.keptnuclei)
    print('Error less than 5cm:', calc.percent, '%')



    with h5py.File('calculated_position-'+filename+'-seed-{}'.format(seed)+'.hdf5', 'w') as f:    
        calc_atom_x = f.create_dataset('calc_atom_x[cm]', data=calc.calculated_atom[:, 0], dtype='float64')
        calc_atom_y = f.create_dataset('calc_atom_y[cm]', data=calc.calculated_atom[:, 1], dtype='float64')
        calc_atom_z = f.create_dataset('calc_atom_z[cm]', data=calc.calculated_atom[:, 2], dtype='float64')
        atom_x = f.create_dataset('atom_x[cm]', data=calc.atom[:, 0], dtype='float64')
        atom_y = f.create_dataset('atom_y[cm]', data=calc.atom[:, 1], dtype='float64')
        atom_z = f.create_dataset('atom_z[cm]', data=calc.atom[:, 2], dtype='float64')
        atom_type = f.create_dataset('atom_type', data=calc.atom[:, 3], dtype='float64')
        num_parts = f.create_dataset('number_of_neutrons', data = calc.originalsamplesize, dtype='float64')
        if commands['--density']:    
            histcount = f.create_dataset('hist_count', data=H, dtype = 'int')
            sidex = f.create_dataset('edge_x', data=edges[0], dtype = 'float64')
            sidey = f.create_dataset('edge_y', data=edges[1], dtype = 'float64')
            sidez = f.create_dataset('edge_z', data=edges[2], dtype = 'float64')

st2 = timeit.default_timer()
print("RUN TIME : {0}".format(st2-st1))
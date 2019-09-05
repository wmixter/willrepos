#!/usr/bin/env python3

"""
Usage:
    calc_nucleus_location  <filename> <num_of_neutrons> [options]
    
Options:
   -h  --help           This information
   -C --carbon-exact             filters only carbon atoms
   -c --carbon
   -r --rate RATE          rate [default: 1e7]
   -s --seed SEED          seed
   -e --errhist               Creates a histogram of the error
   -p --plotall            Plots all data points (warning for large data sets)
   -S --single INDEX            option to analyze a single data point
   -g --gauss STDV        format: alpha.pos[cm], alpha.time[ns], gamma.time[ns], gamma.energy[%]; Adds a gaussian error to all values [default: 0,0,0,0]
   -G --gausspreset       Adds preset values of gaussian error: alpha.pos=0.1cm; alpha.time=1ns; gamma.time=1ns; gamma.energy=7%
   -E --energyhist
   -R --ratesweep RANGE    format: START,FINISH where START/FINISH are exponents of the range values. (ex. range from 1 to 1e9; enter 0,9)
   --g-alpha-pos RANGE           format: START,FINISH [cm]
   --g-alpha-time RANGE           format: START,FINISH [ns]
   --g-gamma-time RANGE           format: START,FINISH [ns]
   --g-gamma-energy RANGE           format: START,FINISH [%]
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.constants as spc
from scipy.optimize import fsolve
import timeit
from decimal import Decimal
from docopt import docopt
#There is a problem with the most recent h5py package
#solved by using pip install h5py==2.8.0rc1
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


commands = docopt(__doc__, version='1.0')
print(commands)


filename = commands['<filename>']

st1 = timeit.default_timer()

if commands['--seed']:
    seed = int(commands['--seed'])
    print('Seed is:',seed)
else:
    seed = np.random.randint(100000,size=1)[0]
    print('Seed is:',seed)


with h5py.File(filename, 'r') as hdf:
    #ls = list(hdf.keys()) 
    #print('list of dataset in this file: \n', ls)

    #Create numpy array for each dataset
    #comment out any unwanted datasets
   # all_data = np.array(hdf.get('all_data'))
    original_alpha_y = np.array(hdf.get('alpha_y[cm]'))
    original_alpha_x = np.array(hdf.get('alpha_x[cm]'))
    original_alpha_t = np.array(hdf.get('alpha_t[s]'))
    original_atom_type = np.array(hdf.get('atom_type'))
    original_atom_x = np.array(hdf.get('atom_x[cm]'))
    original_atom_y = np.array(hdf.get('atom_y[cm]'))
    original_atom_z = np.array(hdf.get('atom_z[cm]'))
    original_gamma_t = np.array(hdf.get('gamma_t[s]'))
    original_gamma_e = np.array(hdf.get('gamma_e[MeV]'))
    original_gamma_x = np.array(hdf.get('gamma_x[cm]'))
    original_gamma_y = np.array(hdf.get('gamma_y[cm]'))
    original_gamma_z = np.array(hdf.get('gamma_z[cm]'))
hdf.close()
datasize = len(original_alpha_y)
#print('max gamma_t is:', max(gamma_t))
#print('min gamma_t is:', min(gamma_t))
original_alpha_z = -6*np.ones_like(original_alpha_y) #locatino of detector
print('alpha xy', original_alpha_x[53],original_alpha_y[53] )
print('alpha t, gamma t', original_alpha_t[53], original_gamma_t[53])
if commands['--gausspreset']:
    g_a_pos = 0.1 # error of 1mm
    g_a_time = 1*1e-9 # error of 1ns
    g_g_time = 1*1e-9 # error of 1ns
    g_g_e = .07*4.439 # error of 7%
else:
    gscale = commands['--gauss']
    gscale = gscale.split(',')
    g_a_pos = [float(gscale[0])] #gaussian error scale for alpha position
    g_a_time = [float(gscale[1])*1e-9] #gaussian error scale for alpha time
    g_g_time = [float(gscale[2])*1e-9] #gaussian error scale for gamma time
    g_g_e = [(float(gscale[3])*4.439)/100] #gaussian error scale for gamma energy
RN_gauss = np.random.RandomState(seed)


        
#Set the point location of the gamma detector
original_gamma_x = np.ones_like(original_gamma_x)*24.65 #no error
original_gamma_y = np.zeros_like(original_gamma_y) #no error
original_gamma_z = np.ones_like(original_gamma_z)*50 #no error



if not commands['--g-alpha-pos'] and not commands['--g-alpha-time'] and not commands['--g-gamma-time'] and not commands['--g-gamma-energy']:
    print('Gaussian error scale is:')
    print('\t alpha pos:',g_a_pos,'cm','\t alpha time:',g_a_time,'ns')
    print('\t gamma energy:',g_g_e,'MeV','\t gamma time:',g_g_time,'ns')

if commands['--g-alpha-pos']:
    g_a_pos = commands['--g-alpha-pos']
    g_a_pos = g_a_pos.split(',')
    start = float(g_a_pos[0])
    stop = float(g_a_pos[1])
    print('alpha pos. error sweep from {} to {} cm'.format(start,stop))
    g_a_pos = np.linspace(start,stop,num=50)
else:
    g_a_pos = [0]
if commands['--g-alpha-time']:
    g_a_time = commands['--g-alpha-time']
    g_a_time = g_a_time.split(',')
    start = float(g_a_time[0])
    stop = float(g_a_time[1])
    print('alpha time error sweep from {} to {} ns'.format(start,stop))
    g_a_time = np.linspace(start,stop,num=50)
else:
    g_a_time = [0]
if commands['--g-gamma-time']:
    g_g_time = commands['--g-gamma-time']
    g_g_time = g_g_time.split(',')
    start = float(g_g_time[0])
    stop = float(g_g_time[1])
    print('gamma time error sweep from {} to {} ns'.format(start,stop))
    g_g_time = np.linspace(start,stop,num=50)
else:
    g_g_time = [0]
if commands['--g-gamma-energy']:
    g_g_e = commands['--g-gamma-energy']
    g_g_e = g_g_e.split(',')
    start = float(g_g_e[0])
    stop = float(g_g_e[1])
    print('gamma energy error sweep from {}% to {}%'.format(start,stop))
    g_g_e = np.linspace(start,stop,num=50)
else:
    g_g_e = [0]


def select_alpha(indices):
    global alpha_y, alpha_x, alpha_z, alpha_t
    alpha_y = alpha_y[indices]
    alpha_x = alpha_x[indices]
    alpha_z = alpha_z[indices]
    alpha_t = alpha_t[indices]
    global exact_alpha_y, exact_alpha_x, exact_alpha_z, exact_alpha_t
    exact_alpha_y = exact_alpha_y[indices]
    exact_alpha_x = exact_alpha_x[indices]
    exact_alpha_z = exact_alpha_z[indices]
    exact_alpha_t = exact_alpha_t[indices]
    
def select_atom(indices):
    global atom_x, atom_y, atom_z
    atom_x = atom_x[indices]
    atom_y = atom_y[indices]
    atom_z = atom_z[indices]
    global atom_type
    atom_type = atom_type[indices]
    global exact_atom_x, exact_atom_y, exact_atom_z
    exact_atom_x = exact_atom_x[indices]
    exact_atom_y = exact_atom_y[indices]
    exact_atom_z = exact_atom_z[indices]
    
def select_gamma(indices):
    global gamma_t, gamma_e, gamma_x, gamma_y, gamma_z
    gamma_t = gamma_t[indices]
    gamma_e = gamma_e[indices]
    gamma_x = gamma_x[indices]
    gamma_y = gamma_y[indices]
    gamma_z = gamma_z[indices]
    global exact_gamma_t, exact_gamma_e, exact_gamma_x, exact_gamma_y, exact_gamma_z
    exact_gamma_t = exact_gamma_t[indices]
    exact_gamma_e = exact_gamma_e[indices]
    exact_gamma_x = exact_gamma_x[indices]
    exact_gamma_y = exact_gamma_y[indices]
    exact_gamma_z = exact_gamma_z[indices]


def select(indices):
    global alpha_y, alpha_x, alpha_z, alpha_t
    alpha_y = alpha_y[indices]
    alpha_x = alpha_x[indices]
    alpha_z = alpha_z[indices]
    alpha_t = alpha_t[indices]
    global exact_alpha_y, exact_alpha_x, exact_alpha_z, exact_alpha_t
    exact_alpha_y = exact_alpha_y[indices]
    exact_alpha_x = exact_alpha_x[indices]
    exact_alpha_z = exact_alpha_z[indices]
    exact_alpha_t = exact_alpha_t[indices]
    global atom_x, atom_y, atom_z
    atom_x = atom_x[indices]
    atom_y = atom_y[indices]
    atom_z = atom_z[indices]
    global atom_type
    atom_type = atom_type[indices]
    global exact_atom_x, exact_atom_y, exact_atom_z
    exact_atom_x = exact_atom_x[indices]
    exact_atom_y = exact_atom_y[indices]
    exact_atom_z = exact_atom_z[indices]
    global gamma_t, gamma_e, gamma_x, gamma_y, gamma_z
    gamma_t = gamma_t[indices]
    gamma_e = gamma_e[indices]
    gamma_x = gamma_x[indices]
    gamma_y = gamma_y[indices]
    gamma_z = gamma_z[indices]
    global exact_gamma_t, exact_gamma_e, exact_gamma_x, exact_gamma_y, exact_gamma_z
    exact_gamma_t = exact_gamma_t[indices]
    exact_gamma_e = exact_gamma_e[indices]
    exact_gamma_x = exact_gamma_x[indices]
    exact_gamma_y = exact_gamma_y[indices]
    exact_gamma_z = exact_gamma_z[indices]
    


#alpha velocity calculation
MeV_alpha = 3.5 #MeV
energy_alpha = MeV_alpha*spc.physical_constants['electron volt-joule relationship'][0]*(1E6) #Joules
mass_alpha = spc.physical_constants['alpha particle mass'][0]#kg
v_alpha = np.sqrt((2*energy_alpha)/mass_alpha)*100 #cm/s
r_alpha = np.sqrt(original_alpha_x**2 + original_alpha_y**2 + original_alpha_z**2)
calc_t_alpha = r_alpha/v_alpha
#gamma velocity
c = (spc.c)*100 #cm/s

#neutron velocity calculation
MeV_neut = 14.0 #MeV
energy_neut = MeV_neut*spc.physical_constants['electron volt-joule relationship'][0]*(1E6) #Joules
mass_neut = spc.m_n #kg
v_neut = np.sqrt((2*energy_neut)/mass_neut)*100 #cm/s

num_parts = float(commands['<num_of_neutrons>'])

indexge = 0 #run index for gamma energy
carbonsweep = []
sweeperrorge = [] #sweep error gamma energy
for a in g_g_e:
    sweeperrorgt = [] #sweep error gamma position
    for b in g_g_time:
        sweeperrorat = [] #sweep error alpha position
        for c in g_a_time:
            sweeperrorap = [] #sweep error alpha position
            for d in g_a_pos:
                alpha_x = original_alpha_x
                alpha_y = original_alpha_y
                alpha_z = original_alpha_z
                alpha_t = original_alpha_t
                atom_x = original_atom_x
                atom_y = original_atom_y
                atom_z = original_atom_z
                atom_type = original_atom_type
                gamma_e = original_gamma_e
                gamma_t = original_gamma_t
                gamma_x = original_gamma_x
                gamma_y = original_gamma_y
                gamma_z = original_gamma_z
                exact_alpha_x = original_alpha_x
                exact_alpha_y = original_alpha_y
                exact_alpha_z = original_alpha_z
                exact_alpha_t = original_alpha_t
                exact_atom_x = original_atom_x
                exact_atom_y = original_atom_y
                exact_atom_z = original_atom_z
                exact_atom_type = original_atom_type
                exact_gamma_e = original_gamma_e
                exact_gamma_t = original_gamma_t
                exact_gamma_x = original_gamma_x
                exact_gamma_y = original_gamma_y
                exact_gamma_z = original_gamma_z
                gamma_e = gamma_e + (RN_gauss.normal(scale=a, size=len(gamma_e))*4.439)/100
                print('alpha xy', alpha_x[53],alpha_y[53] )
                print('alpha xy', exact_alpha_x[53],exact_alpha_y[53] )
                print('alpha t, gamma t', alpha_t[53], gamma_t[53])
                print('alpha t, gamma t', exact_alpha_t[53], exact_gamma_t[53])
                print('gamma_xyz', gamma_x[53], gamma_y[53], gamma_z[53])
                for i in range(len(gamma_e)):
                    if gamma_e[i]<0:
                        gamma_e[i]=0
                
                gamma_t = gamma_t + RN_gauss.normal(scale=b, size=len(gamma_t))*1e-9
                
                alpha_t = alpha_t + RN_gauss.normal(scale=c, size=len(alpha_t))*1e-9
                
                alpha_y = alpha_y + RN_gauss.normal(scale=d, size=len(alpha_y))
                alpha_z = alpha_z + RN_gauss.normal(scale=d, size=len(alpha_z))
                print('abcd', a, b, c, d)
                if commands['--ratesweep']:
                    rate = commands['--ratesweep']
                    rate = rate.split(',')
                    start = float(rate[0])
                    stop = float(rate[1])
                    print('Number of samples:', datasize)
                    print('Rate sweep from 1E{} to 1E{} gammas detected/second'.format(start,stop))
                    rate = np.logspace(start,stop,num=50)
                else:
                    rate = [float(commands['--rate'])] #events per second
                keptnuclei = []
                for r in rate:
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time) and len(g_g_time)==1 and len(g_g_e)==1:
                        print('Rate is:','%.0E' % Decimal('{}'.format(r)))
                    RN_time = np.random.RandomState(seed)
                    secs = num_parts/r #Number of seconds for this many samples
                    time = np.sort(RN_time.uniform(low=0, high=secs, size=len(alpha_y)))
                    print('time here', time[53])
                    tg = gamma_t+time #this simulates the time gamma is measured
                    ta = alpha_t+time #this simulates the time alpha is measured
                    print('ta,tg', ta[53], tg[53])
                    closegap = 0 #fraction change of gap
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                        print('Number of data points:',datasize)
                    morethanone_g = 0
                    zero_g=0
                    onematch = []
                    mindiff = np.min(tg-ta)
                    maxdiff = np.max(tg-ta)
                    print('here', np.where(alpha_x==-0.080971934))
                    print('alpha xy', alpha_x[53],alpha_y[53] )
                    #Take only alphas that hit detector
                    detectalphax = np.where(np.logical_and(exact_alpha_x<5.2, exact_alpha_x>-5.2))
                    select_alpha(detectalphax)
                    ta = ta[detectalphax]
                    detectalphay = np.where(np.logical_and(exact_alpha_y<5.2, exact_alpha_y>-5.2))
                    select_alpha(detectalphay)
                    ta = ta[detectalphay]
                    print('here1.5', np.where(alpha_x==-0.080971934))
                    print('t here 1.5', np.where(ta==1.1512624253118975))
                    index = 0
                    print('min/max', mindiff, maxdiff)
                # =============================================================================
                #     num_alphas = 0.05*num_parts - len(alpha_y)
                #     timealpha = np.sort(RN_time.uniform(low=0, high=secs, size=int(num_alphas)))
                #     print('num_alphas', num_alphas)
                #     print('timealpha', timealpha)
                # =============================================================================
                    correlated_alpha = []
                    for v in ta:
                        lowerlimit = v + mindiff*(1+closegap)
                        upperlimit = v + maxdiff*(1-closegap)
                        lowerindex = np.searchsorted(tg,lowerlimit)
                        upperindex = np.searchsorted(tg,upperlimit, side='right')
                        possiblegamma = upperindex-lowerindex
                        if possiblegamma>1:
                            morethanone_g += 1
                        elif possiblegamma==0:
                            zero_g += 1
                        elif possiblegamma==1:
                            onematch.append(lowerindex)
                            correlated_alpha.append(index)
                        index +=1
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                        print('More than one gamma per neutron:',morethanone_g)
                        print('No gamma per neutron:',zero_g)
                    onematch = np.array(onematch).flatten()
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                        if len(onematch)==0:
                            print('ERROR: No gamma-alpha matches were found')
                            quit()
                    tg = tg[onematch]
                    ta = ta[correlated_alpha]
                    select_gamma(onematch)
                    select_atom(onematch)
                    print('length of onematch', len(onematch))
                    print('length of gamma', len(alpha_y))
                    select_alpha(correlated_alpha)
                    print('length of correlated alpha', len(correlated_alpha))
                    print('here2', np.where(alpha_x==-0.080971934))
                    print('t here 2', np.where(ta==1.1512624253118975))
                    print('alpha xy', alpha_x[45],alpha_y[45] )
                    print('ta,tg', ta[45], tg[45])
                    print('gamma_xyz', gamma_x[45], gamma_y[45], gamma_z[45])
                    morethanone_n = 0
                    zero_n=0
                    onematch = []
                    correlated_gamma = []
                    index = 0
                    for v in tg:
                        lowerlimit = v - maxdiff*(1+closegap)
                        upperlimit = v - mindiff*(1-closegap)
                        lowerindex = np.searchsorted(ta,lowerlimit)
                        upperindex = np.searchsorted(ta,upperlimit, side='right')
                        possibleneut = upperindex-lowerindex
                        if possibleneut>1:
                            morethanone_n += 1
                        elif possibleneut==0:
                            zero_n +=1
                        elif possibleneut==1:
                            onematch.append(lowerindex)
                            correlated_gamma.append(index)
                        index +=1
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                        print('More than one neutrona per gamma:',morethanone_n)
                        print('No neutron per gamma:',zero_n)
                    onematch = np.array(onematch).flatten()
                    tg = tg[correlated_gamma]
                    ta = ta[onematch]
                    calc_t_alpha = calc_t_alpha[onematch]
                    select_alpha(onematch)
                    print('here3', np.where(alpha_x==-0.080971934))

                    select_gamma(correlated_gamma)
                    select_atom(correlated_gamma)
                    print('t here 3', np.where(ta==1.1512624253118975))
                    print('ta,tg', ta[45], tg[45])
                    dt = ta - calc_t_alpha #calculate what time the alpha was born
                    tt = tg - dt #this is the calculated flight time of the gamma and neutron
                # =============================================================================
                #     print('alpha length',len(alpha_y))
                #     print('gamma length',len(gamma_y))
                # =============================================================================
                    keptnuclei.append((len(alpha_y)/datasize)*100)
                    if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                        print('Percent of nuclei kept:',(len(alpha_y)/datasize)*100, '%')
            
                if commands['--carbon']:
                    lowerlimit = 4.439-g_g_e[indexge]-.0001 #MeV
                    upperlimit = 4.439+g_g_e[indexge]+.0001 #MeV
                    
                    carbon_est = []
                    notcarbon = 0
                    for i in range(len(gamma_e)):
                        if gamma_e[i]>=lowerlimit and gamma_e[i]<=upperlimit:
                            carbon_est.append(i)
                        else:
                            notcarbon += 1
                    if len(g_g_e)==1:
                        print('Carbon range is between {} and {} MeV'.format(lowerlimit,upperlimit))
                        print('Number of possible carbon:', len(carbon_est))
                        print('Number of other nuclei:', notcarbon)
                    select(carbon_est)
                    tt = tt[carbon_est]
                    carbonsweep.append(len(carbon_est))
                    
                
                if commands['--carbon-exact']:
                    carbon = np.where(atom_type==6012)
                    select(carbon)
                    tt = tt[carbon]
                    print('Exact number of Carbon nuclei:',len(alpha_y))
                print('THIS SHOULD MATCH', len(alpha_y), len(gamma_x))
                
                print('t here 4', np.where(ta==1.1512624253118975))
                print('ta,tg', ta[45], tg[45])
                print('gamma_xyz', gamma_x[45], gamma_y[45], gamma_z[45])
                #Unit vector components for neutron velocity
                vmag = np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2)
                vx = -alpha_x/vmag
                vy = -alpha_y/vmag
                vz = -alpha_z/vmag
                print('vx', vx[0:15])
                print('tt', tt[0:15])
                print('vneut', v_neut)
                print('gx', gamma_x[0:15])
                print('gy', gamma_y[0:15])
                print('gz', gamma_z[0:15])
                cx = np.ones_like(vx)
                cy = np.ones_like(vx)
                cz = np.ones_like(vx)
                tn = np.ones_like(vx)
                tg = np.ones_like(vx)
                for j in range(len(vx)):
                    def equations(p):
                        Cx, Cy, Cz, Tn, Tg = p
                        return (Tn*v_neut*vx[j] - Cx, Tn*v_neut*vy[j] - Cy, Tn*v_neut*vz[j] - Cz, Tg+Tn-tt[j], np.sqrt((Cx-gamma_x[j])**2 + (Cy-gamma_y[j])**2 + (Cz-gamma_z[j])**2)/c - Tg)
                    if j == 0:
                        Cx, Cy, Cz, Tn, Tg = fsolve(equations, (1, 1, 1, 1, 1))
                    else:
                        Cx, Cy, Cz, Tn, Tg = fsolve(equations, (Cx, Cy, Cz, Tn, Tg))
                    cx[j] = Cx
                    cy[j] = Cy
                    cz[j] = Cz
                    tn[j] = Tn
                    tg[j] = Tg
                errorx = ((cx-exact_atom_x)/exact_atom_x)*100
                errory = ((cy-exact_atom_y)/exact_atom_y)*100
                errorz = ((cz-exact_atom_z)/exact_atom_z)*100
                errordistance = np.sqrt((cx-exact_atom_x)**2 + (cy-exact_atom_y)**2 + (cz-exact_atom_z)**2)
                lessthan5 = [a for a in errordistance if a < 5]
                if len(errordistance) != 0:
                    percent = len(lessthan5)/len(errordistance)*100
                else:
                    percent = 0
                if len(rate)==1 and len(g_a_pos)==1 and len(g_a_time)==1 and len(g_g_time)==1 and len(g_g_e)==1:
                    print('Error less than 5cm:',percent,'%')
                sweeperrorap.append(percent)
            sweeperrorat.append(percent)
        sweeperrorgt.append(percent)
    sweeperrorge.append(percent)
    
#with h5py.File(filename+'calculated_position'+'-seed-{}'.format(seed) + '.hdf5', 'w') as f:
with h5py.File(filename+'-calculated_position'+'.hdf5', 'w') as f:    
    calc_atom_x = f.create_dataset('calc_atom_x[cm]', data=cx, dtype='f')
    calc_atom_y = f.create_dataset('calc_atom_y[cm]', data=cy, dtype='f')
    calc_atom_z = f.create_dataset('calc_atom_z[cm]', data=cz, dtype='f')
    calc_neutron_time = f.create_dataset('calc_neutron_time[s]', data=tn, dtype='f')
    calc_gamma_time = f.create_dataset('calc_gamma_time[s]', data=tg, dtype='f')
f.close()

st2 = timeit.default_timer()

if commands['--single']:
    #index for reporting
    i = int(commands['--single'])
    print('index is', i)
    print('exact atom location is:', atom_x[i], atom_y[i], atom_z[i], 'cm')
    print('calculated atom location is:', cx[i], cy[i], cz[i], 'cm')
    print('neutron flight time is:', tn[i], 's')
    print('gamma flight time is:', tg[i], 's')
    print('error distance is', errordistance[i], 'cm')
    plt.close()
    fig2 = plt.figure(2)
    axis = fig2.gca(projection='3d')
    axis.set_proj_type('ortho') #make z-axis orthogonal

    axis.scatter(exact_atom_x[i], exact_atom_y[i], exact_atom_z[i], c='orange', s=10, label='nuclei')
    axis.scatter(exact_alpha_x[i], exact_alpha_y[i], exact_alpha_z[i], c='yellow', s=10, label='alpha')
    axis.scatter(exact_gamma_x[i], exact_gamma_y[i], exact_gamma_z[i], c='red', s=10, label='gamma detected')

    nx = np.zeros_like(atom_x)
    ny = np.zeros_like(atom_x)
    nz = np.zeros_like(atom_x)
    axis.scatter(nx[i], ny[i], nz[i], c='green', s=10, label='origin')



    axis.plot([exact_alpha_x[i], nx[i]], [exact_alpha_y[i], ny[i]], [exact_alpha_z[i], nz[i]], label='origin')

    axis.plot([nx[i], cx[i]], [ny[i], cy[i]], [nz[i], cz[i]], color='violet', label='calculated')
    axis.scatter(cx[i], cy[i], cz[i], c='violet', s=10, label='calculated')

    plt.title('Gammas from Carbon graph from {}'.format(filename))
    axis.set_xlim([-10, 120])
    axis.set_ylim([-40, 40])
    axis.set_zlim([-40, 40])

    plt.legend(loc='upper left')

    axis.set_xlabel('x (cm)')
    axis.set_ylabel('y')
    axis.set_zlabel('z (cm)')
    axis.xaxis.labelpad = 10 #move xlabel downward

    #axis view
    #axis.view_init(0, -90)
    plt.draw()
    plt.show()
    


if commands['--errhist']:
    fig1 = plt.figure(1)
    #Error Histogram
    bins = 100

    plt.hist(errordistance, bins, histtype='step', log=True)
    #n, bins, patches = plt.hist(u, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('error distance [cm]')
    #plt.ylabel('counts')
    plt.title('Error distance from {}. Percent less than 5cm = {}'.format(filename, percent))
    #plt.set_xlim([0,10])
    plt.axvline(x=5, color='red')
    
    plt.savefig('Error_hist-'+filename+'.png')
    #plt.show()
    


if commands['--plotall']:
    fig3 = plt.figure(3)
    axis3 = fig3.gca(projection='3d')
    axis3.set_proj_type('ortho') #make z-axis orthogonal
    
    axis3.scatter(exact_atom_x, exact_atom_y, exact_atom_z, c='blue', s = 10, label='exact nuclei')
    axis3.scatter(exact_alpha_x, exact_alpha_y, exact_alpha_z, c='cyan', s = 10, label='alpha')
    axis3.scatter(exact_gamma_x, exact_gamma_y, exact_gamma_z, c='yellow', s = 10, label='gamma detected')
    axis3.scatter(gamma_x[0],gamma_y[0],gamma_z[0], c='green', s=50, label='Point detector location')
    
    nx = np.zeros_like(exact_atom_x)
    ny = np.zeros_like(exact_atom_x)
    nz = np.zeros_like(exact_atom_x)
    axis3.scatter(nx, ny, nz, c='orange', s = 10, label='origin')
    
    
    
    #axis3.plot([alpha_x,nx], [alpha_y,ny], [alpha_z,nz], label='origin')
    
    #axis3.plot([nx, cx], [ny, cy], [nz, cz], color='violet', label='calculated')
    axis3.scatter(cx, cy, cz, c='red', s = 10, label='calculated nuclei')
    
    plt.title('Gammas from Carbon graph from {}'.format(filename))
    axis3.set_xlim([-40,40])
    axis3.set_ylim([-40,40])
    axis3.set_zlim([-10,120])
    
    plt.legend(loc='upper left')
    
    axis3.set_xlabel('x (cm)')
    axis3.set_ylabel('y')
    axis3.set_zlabel('z (cm)')
    axis3.xaxis.labelpad = 10 #move xlabel downward
    
    #axis view
    axis3.view_init(0, 90)
    plt.draw()
    plt.savefig('plotall-'+filename+'.png')
    plt.show()
if commands['--energyhist']:
    fig4 = plt.figure(4)
    bins = 100
    #energy = [energy[a] for a in range(len(energy)) if energy[a]<10]
    plt.hist(gamma_e, bins,histtype='step')
    plt.xlabel('energy')
    #plt.ylabel('counts')
    plt.title('Energy spectrum at gamma detector from {}'.format(filename))
    #plt.set_xlim([0,10])
    plt.savefig('energyhist-'+filename+'.png')
    #plt.show()

if commands['--ratesweep']:
    fig4 = plt.figure(4)
    plt.semilogx(rate,keptnuclei)
    plt.scatter(rate,keptnuclei, marker='.')
    plt.title('Gamma/Alpha matches versus rate')
    plt.xlabel('rate [gammas detected/second]')
    plt.ylabel('gamma/alpha matches to total neutrons generated [%]')
    plt.savefig('ratesweep-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()
    
    
if commands['--g-alpha-pos']:
    fig5 = plt.figure(5)
    plt.plot(g_a_pos,sweeperrorap)
    plt.scatter(g_a_pos,sweeperrorap, marker='.')
    plt.title('% with error less than 5cm versus alpha position error')
    plt.xlabel('Stdv. of gaussian error in alpha position [cm]')
    plt.ylabel('Percent less than 5cm error [%]')
    plt.savefig('gauss-sweep-alpha_pos-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()

if commands['--g-alpha-time']:
    fig6 = plt.figure(6)
    plt.plot(g_a_time,sweeperrorat)
    plt.scatter(g_a_time,sweeperrorat, marker='.')
    plt.title('% with error less than 5cm versus alpha time error')
    plt.xlabel('Stdv. of gaussian error in alpha time [ns]')
    plt.ylabel('Percent less than 5cm error [%]')
    plt.ylim([0,100])
    plt.savefig('gauss-sweep-alpha_time-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()
    
if commands['--g-gamma-time']:
    fig7 = plt.figure(7)
    plt.plot(g_g_time,sweeperrorgt)
    plt.scatter(g_g_time,sweeperrorgt, marker='.')
    plt.title('% with error less than 5cm versus gamma time error')
    plt.xlabel('Stdv. of gaussian error in gamma time [ns]')
    plt.ylabel('Percent less than 5cm error [%]')
    plt.ylim([0,100])
    plt.savefig('gauss-sweep-gamma_time-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()
    
if commands['--g-gamma-energy']:
    fig8 = plt.figure(8)
    plt.plot(g_g_e,carbonsweep)
    plt.scatter(g_g_e,carbonsweep, marker='.')
    plt.title('Number of carbon atoms found versus gamma energy error')
    plt.xlabel('Stdv. of gaussian error in gamma energy [%]')
    plt.ylabel('Number of carbon atoms found')
    plt.savefig('gauss-sweep-gamma_energy-'+filename+'seed-{}'.format(seed)+'.png')
    #plt.show()
    
print("RUN TIME : {0}".format(st2-st1))


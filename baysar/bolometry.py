import numpy as np
from adas import read_adf11, run_adas406

adf11_dir="/home/adas/adas/adf11/"
hydrogen_adf11_plt=adf11_dir+'plt12/plt12_h.dat'
hydrogen_adf11_prb=adf11_dir+'prb12/prb12_h.dat'

def radiated_power(n0, ni, ne, te, is1, adf11_plt, adf11_prb):
    plt=read_adf11(file=adf11_plt, adf11type='plt', is1=is1, index_1=-1, index_2=-1, te=te, dens=ne) # , all=False, skipzero=False, unit_te='ev')
    prb=read_adf11(file=adf11_prb, adf11type='prb', is1=is1, index_1=-1, index_2=-1, te=te, dens=ne) # , all=False, skipzero=False, unit_te='ev')

    return (n0*ne*plt, ni*ne*prb)

def get_adf11(elem, yr, type, adf11_dir=adf11_dir):
    return adf11_dir+type+str(yr)+'/'+type+str(yr)+'_'+elem.lower()+'.dat'

from baysar.plasmas import get_meta
def impurity_power(nz, tau, ne, te, elem, yr, adf11_plt=None, adf11_prb=None,
                   charge=None, extrapolate=False):

    # get ionisation balance
    meta=get_meta(elem)
    print(yr, elem, te, ne, tau, meta)
    out, _ =run_adas406(year=yr, elem=elem, te=te, dens=ne, tint=tau, meta=meta)

    # get adf11 if not passed
    if adf11_plt is None:
        adf11_plt=get_adf11(elem, yr, type='plt')
    if adf11_prb is None:
        adf11_prb=get_adf11(elem, yr, type='prb')

    # get power!
    power=0
    number_of_electrons=out['ion'].shape[-1]-1
    if charge is None:
        start=0
        end=number_of_electrons
    elif charge is not None:
        start=charge
        if type(extrapolate)==int:
            end=extrapolate+1
        elif extrapolate:
            end=number_of_electrons
        else:
            end=charge+1

    power=0
    number_of_electrons=out['ion'].shape[-1]-1
    for z in np.arange(start, end):
        n0=nz*out['ion'][:, z]
        ni=nz*out['ion'][:, z+1]
        power+=sum( radiated_power(n0, ni, ne, te, z+1, adf11_plt, adf11_prb) )

    return power

hydrogen_adf11_plt=adf11_dir+'plt12/plt12_h.dat'
hydrogen_adf11_prb=adf11_dir+'prb12/prb12_h.dat'

def plasma_power(species, te, ne):
    power=[sum(radiated_power(0, species['main_ion_density'], ne, te, 1, hydrogen_adf11_plt, hydrogen_adf11_prb))]

    # add neutral power
    if 'neutrals' in species:
        for elem in species['neutrals']:
            power.append(sum(radiated_power(species['neutrals'][elem]['dens'], 0, ne, te, 1, hydrogen_adf11_plt, hydrogen_adf11_prb)))

    # add impurity power
    if 'impurities' in species:
        for elemx in species['impurities']:
            elemx_split=elemx.split('_')
            elem=elemx_split[0]
            charge=str(elemx_split[-1])
            tau=species['impurities'][elemx]['tau']
            nz=species['impurities'][elemx]['dens']
            power.append(impurity_power(nz, tau, ne, te, elem, yr=96, adf11_plt=None, adf11_prb=None, charge=None, extrapolate=False))

    return np.array(power)

def bolometry(species, te, ne, length=1):
    return plasma_power(species, te, ne).sum()*length

from baysar.lineshapes import gaussian
from baysar.priors import gaussian_low_pass_cost, gaussian_high_pass_cost
class BolometryChord:
    def __init__(self, signal, error, plasma, length, probtype='likelihood'):
        self.plasma=plasma
        self.length=length
        self.signal=signal
        self.error=error

        self.probtype=probtype
        self.check_init()

    def check_init(self):
        if self.probtype=='likelihood':
            self.peak=self.fit
        elif self.probtype=='prior':
            if not all([len(self.signal)==2 and len(self.error)==2]):
                raise ValueError("not all([len(self.signal)==2 and len(self.error)==2])")
            self.peak=self.prior
        else:
            raise ValueError("type must be in ('likelihood', 'prior')")

        self.forward_model=1e3
        self.peak()

    def __call__(self):
        self.bolometry()
        return self.peak()

    def fit(self):
        # return probability of fit
        return -0.5*np.square( (self.forward_model-self.signal)/self.error )

    def prior(self):
        prob=0
        fs=[gaussian_low_pass_cost, gaussian_high_pass_cost]
        for f, mean, error in zip(fs, self.signal, self.error):
            prob+=f(self.forward_model, mean, error)

        return prob

    def bolometry(self):
        # build species dict
        self.species=self.get_species_dict()
        self.plasma_power=plasma_power(self.species, te, ne)
        self.forward_model=self.plasma_power.sum()*self.length

    def get_species_dict(self):
        self.species=get_species_dict(self.plasma)

def get_species_dict(plasma):
    species={'main_ion_density': plasma.plasma_state['main_ion_density']}
    if plasma.contains_hydrogen:
        species['neutrals']={}
        neutrals=[elem for elem in plasma.species if not any([elem[:1]==(imp+'_')[:1] for imp in plasma.impurities])]
        for elem in neutrals:
            species['neutrals'][elem]={'dens': plasma.plasma_state[elem+'_dens']}

    # for imp in posterior.plasma.impurities:
    #     imp_ions=[elem for elem in posterior.plasma.species if elem[:1]==(imp+'_')[:1]])

    impurities=[elem for elem in plasma.species if any([elem[:1]==(imp+'_')[:1] for imp in plasma.impurities])]
    species['impurities']={}
    for imp in impurities:
        species['impurities'][imp]={'dens': plasma.plasma_state[imp+'_dens'],
                                    'tau': float(plasma.plasma_state[imp+'_tau'])}

    return species

if __name__=='__main__':
    res=20
    ne=1e14+np.zeros(res)
    te=np.logspace(-1, 1, res)
    nz=1e13
    tau=1e-3
    elem='Ne'
    # print( impurity_power(nz, tau, ne, te, elem=elem, yr=96, charge=1) )
    # print( impurity_power(nz, tau, ne, te, elem=elem, yr=96, charge=1, extrapolate=3) )
    # print( impurity_power(nz, tau, ne, te, elem=elem, yr=96, charge=1, extrapolate=True) )
    # print( impurity_power(nz, tau, ne, te, elem=elem, yr=96) )

    # species={}
    # species['main_ion_density']=ne
    # species['neutrals']={'D_0': {'dens': nz}}
    # species['impurities']={'C_1':{'dens': 10e12, 'tau': 1e-3},
    #                        'N_1':{'dens': 5e12, 'tau': 1e-3}}
    # bolo=plasma_power(species, te, ne)

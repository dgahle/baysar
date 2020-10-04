import numpy as np

def gaussian_high_pass_cost(tmp, threshold, error):
    # everything above the threshold is good
    return -0.5*(max([0, threshold-tmp])/error)**2

def gaussian_low_pass_cost(tmp, threshold, error):
    # everything below the threshold is good
    return -0.5*(max([0, tmp-threshold])/error)**2

class AntiprotonCost:
    def __init__(self, plasma, sigma=1e11):
        self.plasma=plasma
        self.anti_profile_varience=sigma

    def __call__(self):
        if any(self.plasma.plasma_state['main_ion_density'] < 0):
            anti_profile=self.plasma.plasma_state['main_ion_density'].clip(max=0)
            return -0.5*np.square(anti_profile/self.anti_profile_varience).sum()
        else:
            return 0.

class MainIonFractionCost:
    def __init__(self, plasma, threshold=0.8, sigma=0.1):
        self.plasma=plasma
        self.threshold=threshold
        self.sigma=sigma

    def __call__(self):
        ne=self.plasma.plasma_state['electron_density'].clip(1)
        n_ion=self.plasma.plasma_state['main_ion_density']

        self.factor=ne.copy()
        self.factor/=self.factor[np.argmax(n_ion)]
        self.factor=self.factor.clip(0.3)

        n_ion_fraction=n_ion/ne

        return sum([gaussian_high_pass_cost(f, k*self.threshold, self.sigma) for f, k in zip(n_ion_fraction, self.factor)])

class StaticElectronPressureCost:
    def __init__(self, plasma, threshold=5e15, sigma=0.1):
        self.plasma=plasma
        self.threshold=threshold
        self.sigma=sigma

    def __call__(self):
        ne=self.plasma.plasma_state['electron_density'].clip(1)
        te=self.plasma.plasma_state['electron_temperature'].clip(.01)
        pe=te*ne

        return sum([gaussian_low_pass_cost(f, 1, self.sigma) for f in pe/self.threshold])

class ElectronDensityTDVCost:
    def __init__(self, plasma, threshold=1, sigma=0.2):
        self.plasma=plasma
        self.threshold=threshold
        self.sigma=sigma

    def __call__(self):
        ne_tdv=abs( np.diff(self.plasma.plasma_state['electron_density']) ) # .sum()
        ni_tdv=abs( np.diff(self.plasma.plasma_state['main_ion_density']) ) # .sum()

        mean=(ne_tdv-ni_tdv)/ni_tdv

        # return sum([gaussian_low_pass_cost(f, 1, self.sigma) for f in mean/self.threshold])
        return gaussian_low_pass_cost(mean.sum()/self.threshold, 1, self.sigma)

class NeutralFractionCost:
    def __init__(self, plasma, threshold=[5e-3, 2e-1], sigma=0.1, species='D_ADAS_0'):
        self.plasma=plasma
        self.threshold=threshold
        self.sigma=sigma
        self.species=species

    def __call__(self):
        n0=self.plasma.plasma_state[self.species+'_dens'].clip(1)
        ne=self.plasma.plasma_state['electron_density']
        f0=n0/ne # .max()

        # low_pass=sum([gaussian_low_pass_cost(f, 1, self.sigma) for f in f0/self.threshold[1]])
        high_pass=sum([gaussian_high_pass_cost(f, 1, self.sigma) for f in f0/self.threshold[0]])

        # return low_pass+high_pass
        return high_pass


def curvature(profile):
    grad1 = np.gradient(profile / max(profile))
    grad2 = np.gradient(grad1)
    return - sum(np.square(grad2) / np.power((1 + np.square(grad1)), 3)) # curvature


class CurvatureCost(object):
    def __init__(self, plasma, scale):
        self.plasma=plasma
        self.scale=scale

        self.los=plasma.profile_function.electron_density.x_points
        self.empty_array=plasma.profile_function.electron_density.empty_theta
        if plasma.profile_function.electron_density.zero_bounds is not None:
            self.slice=slice(1, -1)
        else:
            self.slice=slice(0, len(self.empty_array))

    def __call__(self):
        curves=0
        for tag in ['electron_density']: # , 'electron_temperature']:
            self.empty_array[self.slice]=self.plasma.plasma_theta.get(tag)
            curves+=curvature(self.empty_array)
        return curves*self.scale

from numpy import diff, power, ones, gradient
def flatness_prior(profile, scale=1, x=None):
    if x is not None:
        d=gradient(profile, x)
    else:
        d=gradient(profile)
    # return -abs(scale/profile.max())*abs(diff(profile)).sum()
    return -scale*abs(d).sum()

assert flatness_prior(ones(10), 1)==0, 'Error in flatness_prior'

class FlatnessPriorBasic(object):
    def __init__(self, scale, tag, plasma):
        self.tag=tag
        self.scale=power(10, scale)
        self.plasma=plasma

    def __call__(self):
        return flatness_prior(self.plasma.plasma_state.get(self.tag), self.scale)

class PlasmaGradientPrior:
    def __init__(self, plasma, te_scale, ne_scale):
        self.te_scale=te_scale
        self.ne_scale=ne_scale
        self.plasma=plasma

        self.functions=[self.plasma.profile_function.electron_density, self.plasma.profile_function.electron_temperature]
        self.tags=['electron_density', 'electron_temperature']
        self.scales=[self.ne_scale, self.te_scale]

    def __call__(self):
        ne_x=self.plasma.profile_function.electron_density.x_points
        te_x=self.plasma.profile_function.electron_temperature.x_points
        self.xs=[ne_x, te_x]

        cost=0
        for f, s, x in zip(self.functions, self.scales, self.xs):
            cost+=flatness_prior(f.empty_theta, s, x=x)

        return cost


class FlatnessPrior(object):
    def __init__(self, scale, tag, plasma):
        if tag not in ('electron_density', 'electron_temperature'):
            raise ValueError("tag not in ('electron_density', 'electron_temperature')")

        self.tag=tag
        self.scale=power(10, scale)
        self.plasma=plasma

        if self.tag=='electron_density':
            self.los=plasma.profile_function.electron_density.x_points
            self.empty_array=plasma.profile_function.electron_density.empty_theta
            if plasma.profile_function.electron_density.zero_bounds is not None:
                self.slice=slice(1, -1)
            else:
                self.slice=slice(0, len(self.empty_array))
        elif self.tag=='electron_temperature':
            self.los=plasma.profile_function.electron_temperature.x_points
            self.empty_array=plasma.profile_function.electron_temperature.empty_theta
            if plasma.profile_function.electron_temperature.zero_bounds is not None:
                self.slice=slice(1, -1)
            else:
                self.slice=slice(0, len(self.empty_array))
        else:
            raise ValueError("tag not in ('electron_density', 'electron_temperature')")

    def __call__(self):
        self.empty_array[self.slice]=self.plasma.plasma_theta.get(self.tag)
        return flatness_prior(self.empty_array, self.scale)

class SeparatrixTePrior(object):
    def __init__(self, scale, plasma, index, ne_scale=0):
        self.scale=scale
        self.ne_scale=ne_scale
        self.plasma=plasma
        self.separatrix_position=index

    def __call__(self):
        te=np.array(self.plasma.plasma_theta['electron_temperature'])
        ne=np.array(self.plasma.plasma_theta['electron_density'])
        te_x=self.plasma.profile_function.electron_temperature.x
        ne_x=self.plasma.profile_function.electron_density.x

        te_check=te-te.max()
        ne_check=ne-ne.max()
        # return self.scale*te_check[self.separatrix_position] + self.ne_scale*ne_check[self.separatrix_position]

        te_cost=self.scale*abs(te_x[np.where(te==te.max())]).max()*te_check[self.separatrix_position]
        ne_cost=self.ne_scale*abs(ne_x[np.where(ne==ne.max())]).max()*ne_check[self.separatrix_position]
        return te_cost+ne_cost



class PeakTePrior:
    def __init__(self, plasma, te_min=1, te_err=1):
        self.plasma=plasma
        self.te_min=te_min
        self.te_err=te_err

    def __call__(self):
        te=self.plasma.plasma_state['electron_temperature']
        cost=gaussian_low_pass_cost(te.max(), self.te_min, self.te_err)

        return cost

def get_impurity_species(plasma):
    impurity_species=[]
    tmp={}
    for impurity in plasma.impurities:
        tmp[impurity]=[]
        tmp_ions=[ion for ion in plasma.species if impurity+'_' in ion]

        for ion in tmp_ions:
            impurity_species.append(ion)
            _, charge=plasma.species_to_elem_and_ion(ion)
            tmp[impurity].append(charge)

    return tmp

def tau_difference(plasma, show_taus=False):
    imp_dict=get_impurity_species(plasma)

    taus=[]
    diff_taus=[]
    for imp in imp_dict:
        for ion in imp_dict[imp]:
            tmp_tau_key=imp+'_'+str(ion)+'_tau'
            taus.append(plasma.plasma_state[tmp_tau_key][0])
        for dtau in -np.diff( np.log10(taus[-len(imp_dict[imp]):]) ):
            diff_taus.append(dtau)

    if show_taus:
        print(taus)

    return diff_taus

class TauPrior:
    def __init__(self, plasma, mean=0, sigma=1):
        self.plasma=plasma
        self.mean=mean
        self.sigma=sigma

    def __call__(self):
        self.diff_log_taus=tau_difference(self.plasma)
        logp=np.array([gaussian_low_pass_cost(-dlt, threshold=self.mean, error=self.sigma) for dlt in self.diff_log_taus])
        return logp.sum()

class WallConditionsPrior:
    def __init__(self, plasma, te_min=0.5, ne_min=2e12, te_err=None, ne_err=None):
        self.plasma=plasma
        self.te_min=te_min
        self.ne_min=ne_min
        self.te_err=te_err
        self.ne_err=ne_err

        default_error=.2
        if self.te_err is None:
            self.te_err=default_error
        if self.ne_err is None:
            self.ne_err=default_error

    def __call__(self):
        te=self.plasma.plasma_state['electron_temperature']
        ne=self.plasma.plasma_state['electron_density']

        cost=0
        for t in [te[0], te[-1]]:
            cost+=gaussian_low_pass_cost(t/self.te_min, 1, self.te_err)
        for n in [ne[0], ne[-1]]:
            cost+=gaussian_low_pass_cost(n/self.ne_min, 1, self.ne_err)

        return cost

class SimpleWallPrior:
    def __init__(self, plasma, te_min=0.5, ne_min=2e12, te_err=None, ne_err=None):
        self.plasma=plasma
        self.te_min=te_min
        self.ne_min=ne_min
        self.te_err=te_err
        self.ne_err=ne_err

        default_error=.5
        if self.te_err is None:
            self.te_err=default_error*self.te_min
        if self.ne_err is None:
            self.ne_err=default_error*self.ne_min

    def __call__(self):
        te=self.plasma.plasma_state['electron_temperature'][-1]
        ne=self.plasma.plasma_state['electron_density'][-1]

        cost=0
        cost+=gaussian_low_pass_cost(te, self.te_min, self.te_err)
        cost+=gaussian_low_pass_cost(ne, self.ne_min, self.ne_err)

        return cost

class BowmanTeePrior:
    def __init__(self, plasma, sigma_err=0.2, nu_err=0.2):
        self.plasma=plasma
        self.sigma_err=sigma_err
        self.nu_err=nu_err
        # self.sigma_indicies=[plasma.slices['electron_density'].start+2, plasma.slices['electron_temperature'].start+1]

    def __call__(self):
        cost=0.
        sigma_diff=self.plasma.plasma_theta['electron_temperature'][1]-self.plasma.plasma_theta['electron_density'][2]
        nu_diff=self.plasma.plasma_theta['electron_density'][4]-self.plasma.plasma_theta['electron_temperature'][3]
        cost+=gaussian_low_pass_cost(sigma_diff, 0., self.sigma_err)
        # cost+=gaussian_low_pass_cost(nu_diff, 0., self.nu_err)
        return cost

from numpy import diff

class ChargeStateOrderPrior:
    def __init__(self, posterior, mean=2.0, sigma=0.2):
        self.posterior=posterior
        self.mean=mean
        self.sigma=sigma
        self.impurity_indicies_dict={}
        for counter, c in enumerate(self.posterior.posterior_components[:posterior.plasma.num_chords]):
            for i, l in enumerate(c.lines):
                if 'species' in l.__dict__:
                    if l.species in self.posterior.plasma.impurity_species and not l.species in self.impurity_indicies_dict:
                        self.impurity_indicies_dict[l.species]=(counter, i)

        print(self.impurity_indicies_dict)

    def __call__(self):
        self.history=[]
        tmp=self.impurity_indicies_dict
        tmp1=[]
        cost=0
        for elem in self.posterior.plasma.impurities:
            tmp_history=[]
            for ion in sorted([ion for ion in tmp if ion.startswith(elem)], key=lambda x: tmp[x]):
                chords, line_index=tmp[ion]
                tmp1.append(self.posterior.posterior_components[chords].lines[line_index].ems_te)
                tmp_history.append((elem, ion, tmp1[-1]))
                self.history.append((elem, ion, tmp1[-1]))

            dts=diff([ems_te[-1] for ems_te in sorted(tmp_history, key=lambda x:float(x[1].split('_')[1]))])
            for dt in dts:
                cost+=gaussian_high_pass_cost(dt, self.mean, self.sigma)

        return cost

class WavelengthCalibrationPrior:
    def __init__(self, plasma, mean, std):
        self.plasma_state=plasma.plasma_state
        self.mean=mean
        self.std=std

    def __call__(self):
        cost=0
        if 'calwave_0' in self.plasma_state:
            cwl, disp=self.plasma_state['calwave_0']
            cost+=-0.5*np.square((self.mean-self.plasma_state['calwave_0'])/self.std)

        return cost.sum()


class CalibrationPrior:
    def __init__(self, plasma):
        self.plasma_state=plasma.plasma_state
        self.mean=1
        self.std=0.2

    def __call__(self):
        cost=0
        if 'cal_0' in self.plasma_state:
            cost+=-0.5*np.square((self.mean-self.plasma_state['cal_0'])/self.std)

        return cost.sum()

class GaussianInstrumentFunctionPrior:
    def __init__(self, plasma, mean=0.8, std=0.1):
        self.plasma_state=plasma
        self.mean=mean
        self.std=std

    def __call__(self):
        cost=-0.5*np.square( (self.plasma_state['calint_func_0'][0]-self.mean)/self.std )
        return cost

from numpy import zeros, diag, eye, log, exp, subtract
from scipy.linalg import solve_banded, solve_triangular, ldl

def tridiagonal_banded_form(A):
    B = zeros([3, A.shape[0]])
    B[0, 1:] = diag(A, k = 1)
    B[1, :] = diag(A)
    B[2,:-1] = diag(A, k = -1)
    return B




class SymmetricSystemSolver(object):
    def __init__(self, A):
        L_perm, D, fwd_perms = ldl(A)

        self.L = L_perm[fwd_perms,:]
        self.DB = tridiagonal_banded_form(D)
        self.fwd_perms = fwd_perms
        self.rev_perms = fwd_perms.argsort()

    def __call__(self, b):
        # first store the permuted form of b
        b_perm = b[self.fwd_perms]

        # solve the system by substitution
        y = solve_triangular(self.L, b_perm, lower = True)
        h = solve_banded((1, 1), self.DB, y)
        x_perm = solve_triangular(self.L.T, h, lower = False)

        # now un-permute the solution vector
        x_sol = x_perm[self.rev_perms]
        return x_sol




class GaussianProcessPrior(object):
    def __init__(self, mu, plasma = None, A = 2., L = 0.02, profile=None):
        self.plasma = plasma
        self.tag=profile
        self.A = A
        self.L = L

        self.mu = mu
        self.los=plasma.profile_function.electron_density.x_points
        self.empty_array=plasma.profile_function.electron_density.empty_theta
        if plasma.profile_function.electron_density.zero_bounds is not None:
            self.slice=slice(1, -1)
        else:
            self.slice=slice(0, len(self.empty_array))


        # get the squared-euclidean distance using outer subtraction
        D = subtract.outer(self.los, self.los)**2

        # build covariance matrix using the 'squared-exponential' function
        K = (self.A**2)*exp(-0.5*D/self.L**2)

        # construct the LDL solver for the covariance matrix
        solver = SymmetricSystemSolver(K)

        # use the solver to get the inverse-covariance matrix
        I = eye(K.shape[0])
        self.iK = solver(I)

        covariance_error=abs(self.iK.dot(K)-I).max()
        tolerence=1e-6
        if covariance_error > tolerence:
            # raise ValueError("Error in construction of covariance matrix!"+
            #                 f"max(|iKK-I|) = {covariance_error:g0.3} (> {tolerence})")
            raise ValueError("Error in construction of covariance matrix! \n \
                              max(|iKK-I|) = {} > {}".format(covariance_error, tolerence))

    def __call__(self):
        self.empty_array[self.slice]=self.plasma.plasma_theta.get(self.tag)
        return  self.log_prior( self.empty_array - self.mu )

    def log_prior(self, field):
        return -0.5*(field.T).dot( self.iK.dot(field) )

    # def gradient(self):
    #     grad = zeros(self.plasma.N_params)
    #     grad[self.plasma.slices['Te']] = self.log_prior_gradient( log(self.plasma.get('Te')) - self.mu_Te ) / self.plasma.get('Te')
    #     grad[self.plasma.slices['ne']] = self.log_prior_gradient( log(self.plasma.get('ne')) - self.mu_ne ) / self.plasma.get('ne')
    #     # grad[self.plasma.slices['n0']] = self.log_prior_gradient( log(self.plasma.get('n0')) - self.mu_n0 ) / self.plasma.get('n0')
    #     return grad

    # def log_prior_gradient(self, field):
    #     return -self.iK.dot(field)

if __name__=='__main__':
    pass

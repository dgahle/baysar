import numpy as np

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

from numpy import diff, power, ones
def flatness_prior(profile, scale):
    return -abs(scale/profile.max())*abs(diff(profile)).sum()

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

        self.tags=['electron_density', 'electron_temperature']
        self.scales=[self.ne_scale, self.te_scale]

    def __call__(self):
        cost=0
        for t, s in zip(self.tags, self.scales):
            cost+=flatness_prior(np.array(self.plasma.plasma_theta.get(t)), s)

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
        te_check=te-te.max()
        ne_check=ne-ne.max()
        return self.scale*te_check[self.separatrix_position] + self.ne_scale*ne_check[self.separatrix_position]

class WallConditionsPrior:
    def __init__(self, scale, plasma, ne=None):
        self.scale=scale # power(10, scale)
        self.plasma=plasma
        self.ne_scale=ne

    def __call__(self):
        te=self.plasma.plasma_state['electron_temperature']
        cost=self.scale*(te[0]+te[-1])
        if self.ne_scale is not None:
            ne=np.log10( self.plasma.plasma_state['electron_density'] )
            cost+=self.ne_scale*abs(ne[0]+ne[1])
        return -cost

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

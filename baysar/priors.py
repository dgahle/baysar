import numpy

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

if __name__=='__main__':
    pass

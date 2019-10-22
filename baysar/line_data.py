"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


import os
from numpy import ones

data_pathroot = os.path.expanduser('~/baysar/data/')

# N V 2 TECs down to 1
def add_ion_info(dictionary, wavelenghts, jj_fraction, pecs, exc_block, rec_block, **kwargs):
    keywords = ['n_upper', 'n_lower']
    counter = 0
    for wave, jj_f, file, exc, rec in zip(wavelenghts, jj_fraction, pecs, exc_block, rec_block):
        if file is None:
            file = dictionary['default_pecs']
        dictionary[wave] = {'wavelength': wave, 'jj_frac': jj_f, 'pec': file,
                            'exc_block': exc, 'rec_block': rec}
        for key in keywords:
            if key in kwargs:
                dictionary[wave][key] = kwargs[key][counter]
        counter += 1
    return dictionary

adas_root = '/home/adas/adas/'

adas_line_data = {}

adas_line_data['D'] = {'atomic_mass': 2, 'atomic_charge': 1,
                       'ionisation_balance_year': 12,
                       'ions': ['0']}

adas_line_data['D']['0'] = {'default_pecs': adas_root+'adf15/pec12#h/pec12#h_balmer#h0.dat'}
balmer_pecs=adas_root+'adf15/pec12#h/pec12#h_balmer#h0.dat '

wavelenghts = [3834.34, 3887.99, 3968.99, 4100.61, 4339.28]
jj_fraction = ones(len(wavelenghts)).tolist()
pecs = [balmer_pecs, balmer_pecs, balmer_pecs, balmer_pecs, balmer_pecs]
exc_block = [7, 6, 5, 4, 3]
rec_block = [25, 24, 23, 22, 21]

n_upper = [9, 8, 7, 6, 5]
n_lower = [2, 2, 2, 2, 2]

adas_line_data['D']['0'] = add_ion_info(adas_line_data['D']['0'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block, n_upper=n_upper, n_lower=n_lower)


adas_line_data['C'] = {'atomic_mass': 12, 'atomic_charge': 6,
                       'ionisation_balance_year': 96,
                       'ions': ['1', '2', '3', '4']}

adas_line_data['C']['1'] = {'default_pecs': adas_root+'adf15/pec96#c/pec96#c_vsu#c1.dat'}

wavelenghts = [5268.3, 5124.4, 5143.3]
jj_fraction = [[1], [0.25, 0.25, 0.25, 0.25], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1]] # need to update
pecs = [None, None, None]
exc_block = [24, 15, 12]
rec_block = [74, 65, 62]

adas_line_data['C']['1'] = add_ion_info(adas_line_data['C']['1'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N'] = {'atomic_mass': 14, 'atomic_charge': 7,
                       'ionisation_balance_year': 96,
                       'ions': ['1', '2', '3', '4']}

adas_line_data['N']['1'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_vsu#n1.dat'}

my_n_ii_file = '/home/dgahle/adas/idl_spectra/use_adas208/n_ii_3900_4900_te_1_100.pass'

wavelenghts = [3995, (4026.09, 4039.35), (4035.09, 4041.32, 4043.54, 4044.79, 4056.92),
               (4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08)]
jj_fraction = [[1], [0.944, 0.056], [0.205, 0.562, 0.175, 0.029, 0.029],
               [4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08]]
pecs = [my_n_ii_file, my_n_ii_file, my_n_ii_file, my_n_ii_file]
exc_block = [3, 6, 7, 42]
rec_block = [53, 56, 57, 92]

adas_line_data['N']['1'] = add_ion_info(adas_line_data['N']['1'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['2'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_vsu#n2.dat'}

my_n_iii_file = '/home/dgahle/adas/idl_spectra/use_adas208/n_iii_3900_4900_te_1_100.pass'

wavelenghts = [(3998.63, 4003.58), (4097.33, 4103.34), (4591.98, 4610.55, 4610.74), (4634.14, 4640.64, 4641.85)]
jj_fraction = [[0.375, 0.625], [0.667, 0.333], [0.312, 0.245, 0.443], [0.291, 0.512, 0.197]]
pecs = [my_n_iii_file, my_n_iii_file, my_n_iii_file, my_n_iii_file]
exc_block = [6, 11, 35, 38]
rec_block = [56, 61, 85, 88]

adas_line_data['N']['2'] = add_ion_info(adas_line_data['N']['2'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['3'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_vsu#n3.dat'}

wavelenghts = [4057.76]
jj_fraction = [[1]]
pecs = [None]
exc_block = [17]
rec_block = [58]

adas_line_data['N']['3'] = add_ion_info(adas_line_data['N']['3'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['4'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n4.dat'}

wavelenghts = [(4603.73, 4619.98)]
jj_fraction = [[0.545, 0.455]]
pecs = [None]
exc_block = [25]
rec_block = [72]

adas_line_data['N']['4'] = add_ion_info(adas_line_data['N']['4'], wavelenghts, jj_fraction,
                                        pecs, exc_block, rec_block)

if __name__=='__main__':

    pass

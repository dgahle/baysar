"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


import os
from numpy import ones, arange

data_pathroot = os.path.expanduser('~/baysar/data/')

# N V 2 TECs down to 1
def add_ion_info(dictionary, wavelengths, jj_fraction, pecs, exc_block, rec_block, **kwargs):
    keywords = ['n_upper', 'n_lower']
    counter = 0
    for wave, jj_f, file, exc, rec in zip(wavelengths, jj_fraction, pecs, exc_block, rec_block):
        if file is None:
            file = dictionary['default_pecs']
        dictionary[wave] = {'wavelenght': wave, 'jj_frac': jj_f, 'pec': file,
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

adf11_dir="/home/adas/adas/adf11/"
adas_line_data['D']['plt'] = adf11_dir+'plt12/plt12_h.dat'
adas_line_data['D']['prb'] = adf11_dir+'prb12/prb12_h.dat'

adas_line_data['D']['0'] = {'default_pecs': adas_root+'adf15/pec12#h/pec12#h_pju#h0.dat'}
balmer_pecs=adas_root+'adf15/pec12#h/pec12#h_balmer#h0.dat'

wavelengths = [1215.2, 1025.3, 972.1, 949.3, 937.4,
               930.4, 925.8, 922.8, 920.6, 919.0, 917.7,
               3834.34, 3887.99, 3968.99, 4100.61,
               4339.28, 4860.00, 6561.01]
jj_fraction = ones(len(wavelengths)).tolist()
pecs = [None, None, None, None, None,
        None, None, None, None, None, None,
        balmer_pecs, balmer_pecs, balmer_pecs, balmer_pecs,
        balmer_pecs, balmer_pecs, balmer_pecs]
exc_block = [1, 2, 4, 7, 11,
             16, 22, 29, 37, 46, 56,
             7, 6, 5, 4, 3, 2, 1]
rec_block = [67, 68, 70, 73, 77,
             82, 88, 95, 103, 112, 122,
             25, 24, 23, 22, 21, 20, 19]

n_upper = [2, 3, 4, 5, 6,
           7, 8, 9, 10, 11, 12,
           9, 8, 7, 6, 5, 4, 3]
n_lower = [1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1,
           2, 2, 2, 2, 2, 2, 2]

adas_line_data['D']['0'] = add_ion_info(adas_line_data['D']['0'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block, n_upper=n_upper, n_lower=n_lower)

adas_line_data['D_ADAS'] = {'atomic_mass': 2, 'atomic_charge': 1,
                       'ionisation_balance_year': 12,
                       'ions': ['0']}

adas_line_data['D_ADAS']['0'] = {'default_pecs': adas_root+'adf15/pec12#h/pec12#h_balmer#h0.dat'}
# note only Balmer wavelengths at the adf15 values...
# k=0.
# k=0.14
# k=0.1955
# wavelengths = [1215.2, 1025.3, 972.1, 949.3, 937.4,
#                930.4, 925.8, 922.8, 920.6, 919.0, 917.7,
#                3834.9, 3888.5, 3969.5+.13, 4101.2+.141,
#                4339.9, 4860.6, 6561.9]

wavelengths = [1215.2, 1025.3, 972.1, 949.3, 937.4,
              930.4, 925.8, 922.8, 920.6, 919.0, 917.7,
              3834.9, 3888.5, 3969.5, 4101.2,
              4339.9, 4860.6, 6561.9]

# 6561.9      N= 3 - N= 2    EXCIT
# C      2.          4860.6      N= 4 - N= 2    EXCIT
# C      3.          4339.9      N= 5 - N= 2    EXCIT
# C      4.          4101.2      N= 6 - N= 2    EXCIT
# C      5.          3969.5      N= 7 - N= 2    EXCIT
# C      6.          3888.5      N= 8 - N= 2    EXCIT
# C      7.          3834.9      N= 9 - N= 2    EXCIT
# C      8.          3797.4      N=10 - N= 2    EXCIT
# C      9.          3770.1      N=11 - N= 2    EXCIT

adas_line_data['D_ADAS']['0'] = add_ion_info(adas_line_data['D_ADAS']['0'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block, n_upper=n_upper, n_lower=n_lower)

adas_line_data['H'] = {'atomic_mass': 1, 'atomic_charge': 1,
                       'ionisation_balance_year': 12,
                       'ions': ['0']}

adas_line_data['H']['0'] = {'default_pecs': adas_root+'adf15/pec12#h/pec12#h_balmer#h0.dat'}

wavelengths = [1215.2, 1025.3, 972.1, 949.3, 937.4,
               930.4, 925.8, 922.8, 920.6, 919.0, 917.7,
               3835.40, 3889.06, 3970.01, 4101.73,
               4340.47, 4861.35, 6562.79]

adas_line_data['H']['0'] = add_ion_info(adas_line_data['H']['0'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block, n_upper=n_upper, n_lower=n_lower)

adas_line_data['He'] = {'atomic_mass': 4, 'atomic_charge': 2,
                       'ionisation_balance_year': 96,
                       'ions': ['1']}

adas_line_data['He']['1'] = {'default_pecs': adas_root+'adf15/pec96#he/pec96#he_pju#he1.dat'}

wavelengths = [4685.80]
jj_fraction = [ [1] ] # need to update
pecs = [None]
exc_block = [8]
rec_block = [17]

adas_line_data['He']['1'] = add_ion_info(adas_line_data['He']['1'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['B'] = {'atomic_mass': 10, 'atomic_charge': 5,
                       'ionisation_balance_year': 96,
                       'ions': ['1']}

# (base) dgahle@freia016:~> ls /home/dgahle/Downloads/b96_prototype/
# acd96_b.dat       plt96_b.dat       scd96_b.dat
# pecXXb_pjub1.dat  prb96_b.dat

adas_line_data['B']['scd'] = "/home/dgahle/Downloads/b96_prototype/scd96_b.dat"
adas_line_data['B']['acd'] = "/home/dgahle/Downloads/b96_prototype/acd96_b.dat"
adas_line_data['B']['plt'] = "/home/dgahle/Downloads/b96_prototype/plt96_b.dat"
adas_line_data['B']['prb'] = "/home/dgahle/Downloads/b96_prototype/prb96_b.dat"

my_b_ii_file = "/home/dgahle/Downloads/b96_prototype/pecXXb_pjub1.dat"
wavelengths = [4121.93]
jj_fraction = [[1]]
pecs = [my_b_ii_file]
exc_block = [11]
rec_block = [34]

adas_line_data['B']['1'] = {'default_pecs': my_b_ii_file}
adas_line_data['B']['1'] = add_ion_info(adas_line_data['B']['1'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['C'] = {'atomic_mass': 12, 'atomic_charge': 6,
                       'ionisation_balance_year': 96,
                       'ions': ['1', '2', '3', '4']}

adas_line_data['C']['1'] = {'default_pecs': adas_root+'adf15/pec96#c/pec96#c_vsu#c1.dat'}

my_c_ii_file = '/home/dgahle/adas/idl_spectra/use_adas208/c_ii_vsu.pass'
wavelengths = [(3918.89, 3920.69), 4268.3, (5120.08, 5121.83, 5125.21, 5126.96), (5132.95, 5133.28, 5137.26, 5139.17, 5143.50, 5145.17, 5151.09)]
jj_fraction = [[0.334, 0.666], [1], [0.083, 0.417, 0.333, 0.167], [0.14, 0.151, 0.029, 0.045, 0.139, 0.349, 0.149]]
pecs = [my_c_ii_file, None, None, None]
exc_block = [8, 24, 15, 12]
rec_block = [61, 74, 65, 62]

adas_line_data['C']['1'] = add_ion_info(adas_line_data['C']['1'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['C']['2'] = {'default_pecs': adas_root+'adf15/pec96#c/pec96#c_vsu#c2.dat'}

my_c_iii_file = '/home/dgahle/adas/idl_spectra/use_adas208/c_iii_vsu.pass'
wavelengths = [(4067.94, 4068.92, 4070.26, 4070.31), 4121.85, 4162.90, 4186.90, (4647.42, 4650.25, 4651.47)]
jj_fraction = [[0.225, 0.300, 0.425, 0.050], [1], [1], [1], [0.556, 0.333, 0.111] ] # need to update
# wavelengths = [(4067.94, 4068.92, 4070.26, 4070.31), 4162.90, 4186.90, (4647.42, 4650.25, 4651.47)]
# jj_fraction = [[0.225, 0.300, 0.425, 0.050], [1], [1], [0.556, 0.333, 0.111] ] # need to update
pecs = [my_c_iii_file, my_c_iii_file, my_c_iii_file, my_c_iii_file, None]
exc_block = [5, 6, 7, 9, 2]
rec_block = [39, 40, 41, 43, 52]

adas_line_data['C']['2'] = add_ion_info(adas_line_data['C']['2'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['C']['3'] = {'default_pecs': adas_root+'adf15/pec96#c/pec96#c_vsu#c3.dat'}

my_c_iv_file = '/home/dgahle/adas/idl_spectra/use_adas208/c_iv_vsu.pass'
wavelengths = [4537.80]
jj_fraction = [[1] ] # need to update
pecs = [my_c_iv_file]
exc_block = [1]
rec_block = [3]

adas_line_data['C']['3'] = add_ion_info(adas_line_data['C']['3'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N'] = {'atomic_mass': 14, 'atomic_charge': 7,
                       'ionisation_balance_year': 96,
                       'ions': [str(n) for n in arange(7)]}

adas_line_data['N']['0'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n1.dat'}

adf15_96_vsu_n0="/home/adas/adas/adf15/pec96#n/pec96#n_vsu#n0.dat"

wavelengths = [1200.00, 1160.20, 1134.70, 964.40, 959.70,
               4109.95, 4137.6]
jj_fraction = [[1], [1], [1], [1], [1],
               [1], [1]]
pecs = [None, None, None, None, None,
        adf15_96_vsu_n0, adf15_96_vsu_n0]
exc_block = [1, 2, 3, 4, 5,
             8, 4]
rec_block = [51, 52, 53, 54, 55,
             26, 22]

adas_line_data['N']['0'] = add_ion_info(adas_line_data['N']['0'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['1'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n1.dat'}

my_n_ii_file = '/home/dgahle/adas/idl_spectra/use_adas208/n_ii_3900_4900_te_1_100.pass'

# nist?
# wavelengths = [600.50, 586.30, 669.80, 629.30, 613.70,
#                3919., 3955.85, 3995., (4026.09, 4039.35),
#                (4035.09, 4041.32, 4043.54, 4044.79, 4056.92),
#                (4073.07, 4076.95, 4082.29, 4095.93), 4087.33,
#                (4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08)]


wavelengths = [600.50, 586.30, 669.80, 629.30, 613.70,
               3919., 3955.85, 3995., (4026.09, 4039.35),
               (4035.09, 4041.32, 4043.54, 4044.79, 4056.92),
               (4073.07, 4076.97, 4082.27, 4095.97), 4087.33,
               (4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08)]

jj_fraction = [[1], [1], [1], [1], [1],
               [1], [1], [1], [0.944, 0.056],
               [0.205, 0.562, 0.175, 0.029, 0.029],
               [0.387, 0.1471, 0.329, 0.1369], [1],
               [0.13793103, 0.11206897, 0.06896552, 0.11206897, 0.43103448, 0.13793103]]
pecs = [None, None, None, None, None,
        my_n_ii_file, my_n_ii_file, my_n_ii_file, my_n_ii_file,
        my_n_ii_file, my_n_ii_file, my_n_ii_file, my_n_ii_file]
exc_block = [1, 2, 3, 4, 5,
            1, 2, 3, 6, 7, 9, 10, 42]
rec_block = [61, 62, 63, 64, 65, 51, 52, 53, 56, 57, 59, 60, 92]

adas_line_data['N']['1'] = add_ion_info(adas_line_data['N']['1'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['2'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n2.dat'}

my_n_iii_file = '/home/dgahle/adas/idl_spectra/use_adas208/n_iii_3900_4900_te_1_100.pass'

wavelengths = [991.00, 764.00, 685.70, 452.10, 374.40,
               (3998.63, 4003.58), (4097.33, 4103.43),
               (4591.98, 4610.55, 4610.74), (4634.14, 4640.64, 4641.85)]
jj_fraction = [[1], [1], [1], [1], [1],
               [0.375, 0.625], [0.667, 0.333], [0.312, 0.245, 0.443], [0.291, 0.512, 0.197]]
pecs = [None, None, None, None, None,
        my_n_iii_file, my_n_iii_file, my_n_iii_file, my_n_iii_file]
exc_block = [1, 2, 3, 4, 5, 6, 11, 35, 38]
rec_block = [55, 56, 57, 58, 59, 56, 61, 85, 88]

adas_line_data['N']['2'] = add_ion_info(adas_line_data['N']['2'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['3'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n3.dat'}
n_iv_vsu_pecs=adas_root+'adf15/pec96#n/pec96#n_vsu#n3.dat'
wavelengths = [233.0, 194.3, 180.4, 323.6, 313.8, 4057.76]
jj_fraction = [[1], [1], [1], [1], [1], [1]]
pecs = [None, None, None, None, None, n_iv_vsu_pecs]
exc_block = [1, 2, 3, 4, 5, 17]
rec_block = [62, 63, 64, 65, 66, 58]

adas_line_data['N']['3'] = add_ion_info(adas_line_data['N']['3'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['4'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n4.dat'}

wavelengths = [206.4, 161.8, 147.1, 618.1, 447.2, (4603.73, 4619.98)]
jj_fraction = [[1], [1], [1], [1], [1], [0.545, 0.455]]
pecs = [None, None, None, None, None, None]
exc_block = [1, 2, 3, 4, 5, 25]
rec_block = [48, 49, 50, 51, 52, 72]

adas_line_data['N']['4'] = add_ion_info(adas_line_data['N']['4'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['5'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n5.dat'}

wavelengths = [24.9, 24.9, 23.8, 23.8, 23.3,
               159.3, 159.3, 122.0, 110.0, 174.1]
jj_fraction = [[1], [1], [1], [1], [1],
               [1], [1], [1], [1], [1]]
pecs = [None, None, None, None, None,
        None, None, None, None, None]
exc_block = [n for n in arange(10)+1]
rec_block = [n for n in arange(10)+51]

adas_line_data['N']['5'] = add_ion_info(adas_line_data['N']['5'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['N']['6'] = {'default_pecs': adas_root+'adf15/pec96#n/pec96#n_pju#n6.dat'}

wavelengths = [24.8, 20.9, 19.8, 19.4, 133.8,
               99.1, 88.5, 382.4, 261.4, 826.4]
jj_fraction = [[1], [1], [1], [1], [1],
               [1], [1], [1], [1], [1]]
pecs = [None, None, None, None, None,
        None, None, None, None, None]
exc_block = [n for n in arange(10)+1]
rec_block = [n for n in arange(10)+11]

adas_line_data['N']['6'] = add_ion_info(adas_line_data['N']['6'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['O'] = {'atomic_mass': 16, 'atomic_charge': 8,
                       'ionisation_balance_year': 96,
                       'ions': [str(n) for n in arange(8)]}

my_o_ii_file = '/home/dgahle/adas/idl_spectra/use_adas208/o_ii_vsu.pass'

adas_line_data['O']['1'] = {'default_pecs': my_o_ii_file}

wavelengths = [3973.26]
# wavelengths = [(3954.36, 3973.26, 3982.71), (3967.37, 3985.42, 3992.76, 4007.46),
#                (4069.62, 4069.89, 4072.16, 4075.86, 4078.84, 4084.65, 4085.11, 4092.93, 4094.14, 4107.09),
#                (4084.65, 4096.53), 4112.69, 4113.42, 4152.82, # 404 nm region
#                4563.18, (4590.97, 4595.96, 4596.18), # 465 nm region
#                (4638.86, 4641.81, 4649.13, 4650.84, 4661.63, 4673.73, 4676.24, 4696.35),
#                4700.44, (4690.90, 4691.41, 4701.18, 4701.71),
#                (4698.45, 4699.00, 4703.16, 4705.35, 4741.71), (4699.22, 4705.35), 4710.01]
jj_fraction = [[1]]
# jj_fraction = [ [0., 1.0, 0.], [0.25, 0.25, 0.25, 0.25],
#                 [.1 for l in arange(10)], [.5, .5], [1], [1], [1], # 404 nm region
#                 [1], [0.333, 0.333, 0.333], [0.125 for l in arange(8)], [1], # 465 nm region
#                 [0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.2, 0.2, 0.2], [.5, .5], [1] ]
pecs = [None]
# pecs = [None, None, None, None, None, None, # 404 nm region
#         None, None, None, None, None, None, None, None] # 465 nm region
exc_block = [8]
# exc_block = [8, 9, 10, 11, 12, 13, 14, # 404 nm region
#              26, 27, 28, 29, 30, 31, 32, 33] # 465 nm region
rec_block = [49]
# rec_block = [49, 50, 51, 52, 53, 54, 55, 56, # 404 nm region
#              68, 69, 70, 71, 72, 73, 74, 75] # 465 nm region

adas_line_data['O']['1'] = add_ion_info(adas_line_data['O']['1'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['O']['2'] = {'default_pecs': adas_root+'adf15/pec96#o/pec96#o_vsu#o2.dat'}

wavelengths = [3961.59 ]
jj_fraction = [ [1] ]
pecs = [None ]
exc_block = [16 ]
rec_block = [44 ]

# wavelengths = [3962.69, 3966.79, 4085.40, 4105.01, 4112.81, 4264.14, 4359.08, 4762.61]
# jj_fraction = [ [1], [1], [1], [1], [1], [1], [1], [1] ]
# pecs = [None, None, None, None, None, None, None, None]
# exc_block = [16, 17, 18, 19, 20, 21, 22, 23]
# rec_block = [44, 45, 46, 47, 48, 49, 50, 51]

adas_line_data['O']['2'] = add_ion_info(adas_line_data['O']['2'], wavelengths, jj_fraction,
                                        pecs, exc_block, rec_block)

adas_line_data['Ne'] = {'atomic_mass': 20, 'atomic_charge': 10,
                        'ionisation_balance_year': 96,
                        'ions': [str(n) for n in arange(10)]}

# my_o_ii_file = '/home/dgahle/adas/idl_spectra/use_adas208/o_ii_vsu.pass'

adas_line_data['Ne']['1'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne1.dat'}

wavelengths = [(460.73, 462.39), (454.65, 455.27, 456.28, 456.35, 456.90), (445.04, 446.26, 446.59, 447.82),
               (405.85, 405.85, 407.14), (361.43, 362.46), (357.88, 358.01, 358.14, 358.88, 359.01), 357.54,
               (356.93, 356.13, 355.95), 356.4, (355.66, 355.89, 356.17, 356.88, 357.16),
               (354.98, 355.45, 355.95, 356.44), (327.24, 327.26, 328.09, 328.09), (326.79, 327.63),
               (326.52, 326.54, 327.36), 326.07, 305.2]
jj_fraction = [ [.663, .337], [0.263, 0.35 , 0.019, 0.21 , 0.158], [0.215, 0.323, 0.269, 0.194],
                [0.278, 0.278, 0.444], [.6, .4], [.2, .2, .2, .2, .2], [1], [0.059, 0.471, 0.471], [1],
                [0.620, 0.023, 0.023, 0.310, 0.023], [0.400, 0.133, 0.267, 0.200], [0.067, 0.400, 0.267, 0.267],
                [0.714, 0.286], [0.077, 0.615, 0.308], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [b+1 for b in range(len(jj_fraction))]
rec_block = [b+53 for b in exc_block]

adas_line_data['Ne']['1'] = add_ion_info(adas_line_data['Ne']['1'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)

adas_line_data['Ne']['2'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne2.dat'}

wavelengths = [489.5, 345.8, 323.0, 313.4, 559.4,
               379.3, 340.6, 305.5, 301.1]
jj_fraction = [ [1], [1], [1], [1], [1],
                [1], [1], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [1, 2, 3, 4, 27, 28, 29, 30, 31]
rec_block = [b+52 for b in exc_block]

adas_line_data['Ne']['2'] = add_ion_info(adas_line_data['Ne']['2'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)


adas_line_data['Ne']['3'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne3.dat'}

wavelengths = [542.8, 333.7, 312.2, 469.8, 387.0,
               358.4, 521.8, 421.6, 387.9]
jj_fraction = [ [1], [1], [1], [1], [1],
                [1], [1], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [1, 2, 3, 17, 18, 19, 41, 42, 43]
rec_block = [b+51 for b in exc_block]

adas_line_data['Ne']['3'] = add_ion_info(adas_line_data['Ne']['3'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)


adas_line_data['Ne']['4'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne4.dat'}

wavelengths = [571.0, 482.1, 370.6, 358.9, 330.0,
               686.9, 562.2, 416.2, 401.5, 365.6]
jj_fraction = [ [1], [1], [1], [1], [1],
                [1], [1], [1], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [1, 2, 3, 4, 5, 30, 31, 32, 33, 34]
rec_block = [b+56 for b in exc_block]

adas_line_data['Ne']['4'] = add_ion_info(adas_line_data['Ne']['4'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)


adas_line_data['Ne']['5'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne5.dat'}

wavelengths = [561.4, 434.8, 401.7, 453.3, 386.7, 327.8]
jj_fraction = [ [1], [1], [1], [1], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [1, 2, 3, 29, 30, 31, 51]
rec_block = [b+50 for b in exc_block]

adas_line_data['Ne']['5'] = add_ion_info(adas_line_data['Ne']['5'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)


adas_line_data['Ne']['6'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne6.dat'}

wavelengths = [465.2, 561.6, 486.7, 356.0, 561.3]
jj_fraction = [ [1], [1], [1], [1], [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [1, 16, 17, 18, 47]
rec_block = [b+50 for b in exc_block]

adas_line_data['Ne']['6'] = add_ion_info(adas_line_data['Ne']['6'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)


adas_line_data['Ne']['7'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne7.dat'}

wavelengths = [(770.41, 780.32)]
jj_fraction = [ [0.577, 0.423] ]
pecs = [None for jj in jj_fraction]
exc_block = [1]
rec_block = [51]

adas_line_data['Ne']['7'] = add_ion_info(adas_line_data['Ne']['7'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)



adas_line_data['Ne']['8'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne8.dat'}

wavelengths = [(732.0)]
jj_fraction = [ [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [15]
rec_block = [65]

adas_line_data['Ne']['8'] = add_ion_info(adas_line_data['Ne']['8'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)



adas_line_data['Ne']['9'] = {'default_pecs':  adas_root+'adf15/pec96#ne/pec96#ne_pju#ne9.dat'}

wavelengths = [404.8]
jj_fraction = [ [1] ]
pecs = [None for jj in jj_fraction]
exc_block = [10]
rec_block = [20]

adas_line_data['Ne']['9'] = add_ion_info(adas_line_data['Ne']['9'], wavelengths, jj_fraction,
                                         pecs, exc_block, rec_block)



if __name__=='__main__':
    print('Success!')

    pass

"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


import os

data_pathroot = os.path.expanduser('~/baysar/data/')

line_data = {}

line_data['D'] = {}

line_data['D']['atomic_mass'] = 2
line_data['D']['atomic_charge'] = 1
line_data['D']['ions'] = ['0']

line_data['D']['0'] = {}
line_data['D']['0']['lines'] = ['3968.99', '4100.58']
line_data['D']['0']['3834.34'] = {'wavelength': 3834.34, 'jj_frac': 1, 'n_upper': 9,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_7_3835.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['D']['0']['3887.99'] = {'wavelength': 3887.99, 'jj_frac': 1, 'n_upper': 8,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_6_3889.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['D']['0']['3968.99'] = {'wavelength': 3968.99, 'jj_frac': 1, 'n_upper': 7,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_5_3970.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['D']['0']['4100.61'] = {'wavelength': 4100.61, 'jj_frac': 1, 'n_upper': 6,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_4_4101.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }
line_data['D']['0']['4339.28'] = {'wavelength': 4339.28, 'jj_frac': 1, 'n_upper': 5,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_3_4340.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }

line_data['H'] = {}

line_data['H']['atomic_mass'] = 2
line_data['H']['atomic_charge'] = 1
line_data['H']['ions'] = ['0']

line_data['H']['0'] = {}
line_data['H']['0']['lines'] = ['3968.99', '4100.58']
line_data['H']['0']['3834.34'] = {'wavelength': 3834.34, 'jj_frac': 1, 'n_upper': 9,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_7_3835.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['H']['0']['3887.99'] = {'wavelength': 3887.99, 'jj_frac': 1, 'n_upper': 8,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_6_3889.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['H']['0']['3968.99'] = {'wavelength': 3968.99, 'jj_frac': 1, 'n_upper': 7,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_5_3970.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data['H']['0']['4100.61'] = {'wavelength': 4100.61, 'jj_frac': 1, 'n_upper': 6,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_4_4101.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }
line_data['H']['0']['4339.28'] = {'wavelength': 4339.28, 'jj_frac': 1, 'n_upper': 5,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_3_4340.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }



line_data['C'] = {}

line_data['C']['atomic_mass'] = 12
line_data['C']['atomic_charge'] = 6
line_data['C']['ions'] = ['1', '2']

line_data['C']['1'] = {}
line_data['C']['1']['lines'] = [(4085.0, 4087.0)]

line_data['C']['1'][(4085.0, 4087.0)] = \
    {'wavelength': [4085.0, 4087.0], 'jj_frac': [0.65, 0.35] }


line_data['C']['2'] = {}
line_data['C']['2']['lines'] = [(4647.42, 4650.25, 4651.47)]

line_data['C']['2'][(4647.42, 4650.25, 4651.47)] = \
    {'wavelength': [4647.42, 4650.25, 4651.47], 'jj_frac': [0.2, 0.45, 0.35] }


line_data['N'] = {}

line_data['N']['atomic_mass'] = 14
line_data['N']['atomic_charge'] = 7
line_data['N']['ions'] = ['1', '2', '3']

line_data['N']['1'] = {}
line_data['N']['1']['lines'] = ['3995.0',  '4026.09', '4039.35',
                                '4035.09', '4041.32', '4043.54', '4044.79', '4056.92',
                                '4601.48', '4607.16', '4613.87', '4621.39', '4630.54', '4643.08']

line_data['N']['1']['3995.0'] = {'wavelength': 3995.0,  'jj_frac': 1,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_3_3996.sav' }
line_data['N']['1']['4026.09'] = {'wavelength': 4026.09,  'jj_frac': 0.944,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_6_4031.sav' }
line_data['N']['1']['4039.35'] = {'wavelength': 4039.35, 'jj_frac': 0.056,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_6_4031.sav' }
line_data['N']['1']['4035.09'] = {'wavelength': 4035.09, 'jj_frac': 0.205,
             'tec': data_pathroot + '/n_data/'
                                        'tec_grids/tec_406_grid_n1_7_4042.sav' }
line_data['N']['1']['4041.32'] = {'wavelength': 4041.32, 'jj_frac': 0.562,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_7_4042.sav' }
line_data['N']['1']['4043.54'] = {'wavelength': 4043.54, 'jj_frac': 0.175,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_7_4042.sav' }
line_data['N']['1']['4044.79'] = {'wavelength': 4044.79, 'jj_frac': 0.029,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_7_4042.sav' }
line_data['N']['1']['4056.92'] = {'wavelength': 4056.92, 'jj_frac': 0.029,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_7_4042.sav' }
line_data['N']['1']['4601.48'] = {'wavelength': 4601.48, 'jj_frac': 0.125,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }
line_data['N']['1']['4607.16'] = {'wavelength': 4607.16, 'jj_frac': 0.104,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }
line_data['N']['1']['4613.87'] = {'wavelength': 4613.87, 'jj_frac': 0.067,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }
line_data['N']['1']['4621.39'] = {'wavelength': 4621.39, 'jj_frac': 0.099,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }
line_data['N']['1']['4630.54'] = {'wavelength': 4630.54, 'jj_frac': 0.463,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }
line_data['N']['1']['4643.08'] = {'wavelength': 4643.08, 'jj_frac': 0.142,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n1_42_4624.sav' }

line_data['N']['2'] = {}
line_data['N']['2']['lines'] = ['3998.63', '4003.58', '4097.33', '4103.34',
                                '4591.98', '4610.55', '4610.74',
                                '4634.14', '4640.64', '4641.85']


line_data['N']['2']['3998.63'] = {'wavelength': 3998.63, 'jj_frac': 0.375,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_6_4003.sav' }
line_data['N']['2']['4003.58'] = {'wavelength': 4003.58, 'jj_frac': 0.622,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_6_4003.sav' }
line_data['N']['2']['4097.33'] = {'wavelength': 4097.33, 'jj_frac': 0.667,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_11_4101.sav' }
line_data['N']['2']['4103.34'] = {'wavelength': 4103.34, 'jj_frac': 0.333,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_11_4101.sav' }
line_data['N']['2']['4591.98'] = {'wavelength': 4591.98, 'jj_frac': 0.312,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_35_4604.sav' }
line_data['N']['2']['4610.55'] = {'wavelength': 4610.55, 'jj_frac': 0.245,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_35_4604.sav' }
line_data['N']['2']['4610.74'] = {'wavelength': 4610.74, 'jj_frac': 0.443,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_35_4604.sav' }
line_data['N']['2']['4634.14'] = {'wavelength': 4634.14, 'jj_frac': 0.291,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                         'tec_406_grid_n2_38_4640.sav' }
line_data['N']['2']['4640.64'] = {'wavelength': 4640.64, 'jj_frac': 0.512,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                       '  tec_406_grid_n2_38_4640.sav' }
line_data['N']['2']['4641.85'] = {'wavelength': 4641.85, 'jj_frac': 0.197,
             'tec': data_pathroot + '/n_data/tec_grids/'
                                       'tec_406_grid_n2_38_4640.sav' }

line_data['N']['3'] = {}
line_data['N']['3']['lines'] = ['4057.76', '4606.33']

line_data['N']['3']['4057.76'] = {'wavelength': 4057.76, 'jj_frac': 1,
             'tec': data_pathroot + '/n_data/'
                                         'tec_grids/tec_406_grid_n3_17_4059.sav' }
line_data['N']['3']['4606.33'] = {'wavelength': 4606.33, 'jj_frac': 1 }


line_data['N']['4'] = {}
line_data['N']['4']['lines'] = ['4603.73', '4619.98']

# TODO: Need to upto jj_frac

line_data['N']['4']['4603.73'] = {'wavelength': 4603.73, 'jj_frac': 0.545,
             'tec': data_pathroot + '/n_data/'
                                         'tec_grids/tec_406_grid_n4_25_4610.sav' }
line_data['N']['4']['4619.98'] = {'wavelength': 4619.98, 'jj_frac': 0.455,
             'tec': data_pathroot + '/n_data/'
                                         'tec_grids/tec_406_grid_n4_25_4610.sav' }

line_data_multiplet = {}

line_data_multiplet['D'] = {}

line_data_multiplet['D']['atomic_mass'] = 2
line_data_multiplet['D']['atomic_charge'] = 1
line_data_multiplet['D']['ions'] = ['0']

line_data_multiplet['D']['0'] = {}
line_data_multiplet['D']['0']['lines'] = ['3834.34', '3887.99', '3968.99',
                                          '4100.61', '4339.28']
line_data_multiplet['D']['0']['3834.34'] = {'wavelength': 3834.34, 'jj_frac': 1, 'n_upper': 9,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_7_3835.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['D']['0']['3887.99'] = {'wavelength': 3887.99, 'jj_frac': 1, 'n_upper': 8,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_6_3889.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['D']['0']['3968.99'] = {'wavelength': 3968.99, 'jj_frac': 1, 'n_upper': 7,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_5_3970.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['D']['0']['4100.61'] = {'wavelength': 4100.61, 'jj_frac': 1, 'n_upper': 6,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_4_4101.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }
line_data_multiplet['D']['0']['4339.28'] = {'wavelength': 4339.28, 'jj_frac': 1, 'n_upper': 5,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_3_4340.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }


line_data_multiplet['H'] = {}

line_data_multiplet['H']['atomic_mass'] = 2
line_data_multiplet['H']['atomic_charge'] = 1
line_data_multiplet['H']['ions'] = ['0']

line_data_multiplet['H']['0'] = {}
# line_data_multiplet['H']['0']['lines'] = ['3834.34', '3887.99', '3968.99',
#                                           '4100.61', '4339.28']
line_data_multiplet['H']['0']['3834.34'] = {'wavelength': 3834.34, 'jj_frac': 1, 'n_upper': 9,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_7_3835.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['H']['0']['3887.99'] = {'wavelength': 3887.99, 'jj_frac': 1, 'n_upper': 8,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_6_3889.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['H']['0']['3968.99'] = {'wavelength': 3968.99, 'jj_frac': 1, 'n_upper': 7,
     'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_5_3970.sav',
     'rec_pec': data_pathroot + '/balmer/pec_grid_h0_5_3970.sav',
     'tec': data_pathroot + '/balmer/tec_grids/'
                                 'tec_406_grid_h0_5_3970.sav' }
line_data_multiplet['H']['0']['4100.61'] = {'wavelength': 4100.61, 'jj_frac': 1, 'n_upper': 6,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_4_4101.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }
line_data_multiplet['H']['0']['4339.28'] = {'wavelength': 4339.28, 'jj_frac': 1, 'n_upper': 5,
    'exc_tec': data_pathroot + '/balmer/neutral_exc_tec_406_grid_h0_4_4101.sav',
    'rec_pec': data_pathroot + '/balmer/pec_grid_h0_3_4340.sav',
    'tec': data_pathroot + '/balmer/tec_grids/'
                                'tec_406_grid_h0_4_4101.sav' }



line_data_multiplet['N'] = {}

line_data_multiplet['N']['atomic_mass'] = 14
line_data_multiplet['N']['atomic_charge'] = 7
line_data_multiplet['N']['effective_charge_406'] = data_pathroot + '/n_data/effective_charge_406_grid_n.sav'
line_data_multiplet['N']['ions'] = ['1', '2', '3', '4']

line_data_multiplet['N']['1'] = {}
# line_data_multiplet['N']['1']['lines'] = ['3995.0',  (4026.09, 4039.35),
#                                 (4035.09, 4041.32, 4043.54, 4044.79, 4056.92),
#                                 (4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08)]


line_data_multiplet['N']['1']['3995.0'] = {'wavelength': 3995.0,  'jj_frac': 1,
             'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n1_3_3996.sav' }

line_data_multiplet['N']['1'][(4026.09, 4039.35)] = {'wavelength': [4026.09, 4039.35],  'jj_frac': [0.944, 0.056],
             'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n1_6_4031.sav' }

line_data_multiplet['N']['1'][(4035.09, 4041.32, 4043.54, 4044.79, 4056.92)] = \
            {'wavelength': [4035.09, 4041.32, 4043.54, 4044.79, 4056.92],
             'jj_frac': [0.205, 0.562, 0.175, 0.029, 0.029],
             'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n1_7_4042.sav' }

line_data_multiplet['N']['1'][(4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08)] = \
            {'wavelength': [4601.48, 4607.16, 4613.87, 4621.39, 4630.54, 4643.08],
             'jj_frac': [0.125, 0.104, 0.067, 0.099, 0.463, 0.142],
             'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n1_42_4624.sav' }

# N II 14 TECs down to 4

line_data_multiplet['N']['2'] = {}
# line_data_multiplet['N']['2']['lines'] = [(3998.63, 4003.58), (4097.33, 4103.34),
#                                 (4591.98, 4610.55, 4610.74),
#                                 (4634.14, 4640.64, 4641.85)]


line_data_multiplet['N']['2'][(3998.63, 4003.58)] = \
    {'wavelength': [3998.63, 4003.58], 'jj_frac': [0.375, 0.625],
     'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n2_6_4003.sav'}


line_data_multiplet['N']['2'][(4097.33, 4103.34)] = \
    {'wavelength': [4097.33, 4103.34], 'jj_frac': [0.667, 0.333],
     'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n2_11_4101.sav' }

line_data_multiplet['N']['2'][(4591.98, 4610.55, 4610.74)] = \
    {'wavelength': [4591.98, 4610.55, 4610.74], 'jj_frac': [0.312, 0.245, 0.443],
     'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n2_35_4604.sav' }

line_data_multiplet['N']['2'][(4634.14, 4640.64, 4641.85)] = \
    {'wavelength': [4634.14, 4640.64, 4641.85], 'jj_frac': [0.291, 0.512, 0.197],
     'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n2_38_4640.sav' }

# N III 10 TECs down to 4

line_data_multiplet['N']['3'] = {}
# line_data_multiplet['N']['3']['lines'] = ['4057.76', '4606.33']

line_data_multiplet['N']['3']['4057.76'] = {'wavelength': 4057.76, 'jj_frac': 1,
             'tec': data_pathroot + '/n_data/'
                                         'tec_grids/tec_406_grid_n3_17_4059.sav' }
line_data_multiplet['N']['3']['4606.33'] = {'wavelength': 4606.33, 'jj_frac': 1 }

# No N IV multiplets to change

line_data_multiplet['N']['4'] = {}
# line_data_multiplet['N']['4']['lines'] = [(4603.73, 4619.98)]

# TODO: Need to upto jj_frac

line_data_multiplet['N']['4'][(4603.73, 4619.98)] = {'wavelength': [4603.73, 4619.98], 'jj_frac': [0.545, 0.455],
             'tec': data_pathroot + '/n_data/tec_grids/tec_406_grid_n4_25_4610.sav' }

# N V 2 TECs down to 1

if __name__=='__main__':

    pass


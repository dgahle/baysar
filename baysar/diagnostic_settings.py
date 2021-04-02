_tcv_pupil = (1.50569999217987e2, -0.399749994277954e2)
_tcv_angles = [  '-0.391012817621231',     '-0.377983331680298',     '-0.365014135837555',     '-0.352100670337677',
                 '-0.339238584041595',     '-0.326423466205597',     '-0.313651055097580',     '-0.300917148590088',
                 '-0.288217544555664',     '-0.275548070669174',     '-0.262904673814774',     '-0.250283271074295',
                 '-0.237679839134216',     '-0.225090354681015',     '-0.212510809302330',     '-0.199937224388123',
                 '-0.187365636229515',     '-0.174792051315308',     '-0.162212505936623',     '-0.149623006582260',
                 '-0.137019574642181',     '-0.124398179352283',     '-0.111754782497883',     '-0.0990853235125542',
                 '-0.0863857045769692',     '-0.0736517906188965',     '-0.0608793906867504',     '-0.0480642840266228',
                 '-0.0352021791040897',     '-0.0222887266427279',     '-0.00931951310485601',     '0.00364970066584647']

tcv_pupil = _tcv_pupil
tcv_angles = [float(a) for a in _tcv_angles]

# _mastu_pupil = (1.669e2, -1.6502e2)
from numpy import arange, linspace
mastu_pupil = (1.8e2, -1.55e2)
# mastu_angles = arange(-0.85, -0.15, 0.0127)
mastu_angles = linspace(-0.85, -0.15, 40)

availible_tokamaks = ["TCV", "MASTU", "MAST SXD"]
def get_spectrometer_settings(tokamak):
    # check that tokamak is availible
    if tokamak not in availible_tokamaks:
        raise ValueError(f"{tokamak} not availible! Availible tokamaks are {availible_tokamaks}")

    if tokamak == "TCV":
        return {"pupil": tcv_pupil, "angles": tcv_angles, "instrument_width": 4}

    if tokamak in ["MASTU", "MAST SXD"]:
        return {"pupil": mastu_pupil, "angles": mastu_angles, "instrument_width": 4}

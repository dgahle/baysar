from pathlib import Path
from typing import Union
from numpy import savez
from baysar.solps_spectrometer import SyntheticSpectrometer



def to_dict(self: SyntheticSpectrometer) -> dict:
    attributes_to_save = ['spectra', 'wavelengths']
    return dict([(attr, self[attr]) for attr in attributes_to_save])

def save(self: SyntheticSpectrometer, save_path: Union[Path, str]):
    savez(save_path, **to_dict(self))

def get_synthetic_spectra(mds_number:int, save_path: Union[Path, str]):
    savez(save_path, **to_dict(SyntheticSpectrometer(mds_number)))

if __name__=="__main__":
    from argparse import ArgumentParser

    parser=ArgumentParser()

    parser.add_argument('--mds_number', type=int, default=141893)
    # parser.add_argument('--initial_distribution', type=str, default='output/test_initial_distribution.npy')
    # parser.add_argument('--element', type=str, default='n')
    parser.add_argument('--save', type=str, default=None)
    # parser.add_argument('--plot', action='store_true')

    args=parser.parse_args()

    output_path = Path(__file__).parent / "data"
    if args.save is None: args.save = output_path / f"synthetic_spectra_mds{args.mds_number}"

    get_synthetic_spectra(mds_number=args.mds_number, save_path=args.save)
    spectrometer = SyntheticSpectrometer(args.mds_number)
    # need to get the power weighted Te/ne
    # need to add ems_conc
    def get_line_quanties(spec:SyntheticSpectrometer) -> dict:
        line_quanties = {}
        return line_quanties

    line_quanties = get_line_quanties(spectrometer)




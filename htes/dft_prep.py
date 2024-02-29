from mace.calculators import MACECalculator
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, FIRE
import ase.io, os

mace_model = os.getenv("MACE_MODEL")

def vc_opt(atoms, model = None, device = 'cpu', dispersion = True,
           output_file = None, traj_file = None, fmax = .02,
           opt_algo = FIRE):

    if model is None:
        model = mace_model

    if model is None:
        raise ValueError("Mace model must either be supplied as 'model'"\
                         " kwarg or as MACE_MODEL environment variable")

    calculator = MACECalculator(model_paths=model, device=device,
                                default_dtype = 'float64', dispersion = dispersion)

    atoms.calc = calculator
    filt = UnitCellFilter(atoms)

    # Setup output file for optimization trajectory
    if traj_file is None:
        opt = opt_algo(filt, trajectory='vc_out.traj')
    else:
        opt = opt_algo(filt, trajectory=traj_file)
    opt.run(fmax = fmax)

    if output_file is None:
        atoms.write("vc_atoms.cif")
    else:
        atoms.write(output_file)

def fc_opt(atoms, model = None, device = 'cpu', dispersion = True,
           output_file = None, traj_file = None, fmax = .02,
           opt_algo = FIRE):

    if model is None:
        model = mace_model

    if model is None:
        raise ValueError("Mace model must either be supplied as 'model'"\
                         " kwarg or as MACE_MODEL environment variable")

    calculator = MACECalculator(model_paths=model, device=device,
                                default_dtype = 'float64', dispersion = dispersion)

    atoms.calc = calculator

    # Setup output file for optimization trajectory
    if traj_file is None:
        opt = opt_algo(atoms, trajectory='fc_out.traj')
    else:
        opt = opt_algo(atoms, trajectory=traj_file)
    opt.run(fmax = fmax)

    if output_file is None:
        atoms.write("fc_atoms.cif")
    else:
        atoms.write(output_file)

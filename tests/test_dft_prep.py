from htes.dft_prep import vc_opt, fc_opt
import ase.io

atoms = ase.io.read("data/cspbi3.cif")

#SPECIFY YOUR MACE MODEL IF IT'S NOT AN ENVIRONMENT VARIABLE HERE!
mace_model = None

def test_fc_opt():
    test_atoms = atoms.copy()
    fc_opt(test_atoms, mace_model)

def test_vc_opt():
    test_atoms = atoms.copy()
    vc_opt(test_atoms, mace_model)




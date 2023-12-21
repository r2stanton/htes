from htes.data import sort_by_element, r_z_to_traj
import numpy as np

F = np.load("data/F.npy")
R = np.load("data/R.npy")
Q = np.load("data/Q.npy")
Z = np.load("data/Z.npy")

nZ, nQ, nF, nR = sort_by_element([Z, Q, F, R])

def test_sorting_shapes():
    assert nZ.shape == Z.shape, f"Z shape changed {nZ.shape} vs. {Z.shape}"
    assert nQ.shape == Q.shape, f"Q shape changed {nQ.shape} vs. {Q.shape}"
    assert nF.shape == F.shape, f"F shape changed {nF.shape} vs. {F.shape}"
    assert nR.shape == R.shape, f"R shape changed {nR.shape} vs. {R.shape}"

def test_sorting_numbers():
    nZz = np.count_nonzero(nZ)
    Zz = np.count_nonzero(Z)
    assert nZz == Zz, f"Number of zeroes changed {nZz} vs. {Zz}"

    nQz = np.count_nonzero(nQ)
    Qz = np.count_nonzero(Q)
    assert nQz == Qz, f"Number of zeroes changed {nQz} vs. {Qz}"

    nFz = np.count_nonzero(nF)
    Fz = np.count_nonzero(F)
    assert nFz == Fz, f"Number of zeroes changed {nFz} vs. {Fz}"

    nRz = np.count_nonzero(nR)
    Rz = np.count_nonzero(R)
    assert nRz == Rz, f"Number of zeroes changed {nRz} vs. {Rz}"

def test_sorting_ids():
    nZflat = nZ.flatten()
    Zflat = Z.flatten()

    nZ0 = len(np.where(nZflat == 0)[0])
    Z0 = len(np.where(Zflat == 0)[0])
    assert nZ0 == Z0, f"Number of zeroes changed {nZ0} vs. {Z0}"

    nZ1 = len(np.where(nZflat == 1)[0])
    Z1 = len(np.where(Zflat == 1)[0])
    assert nZ1 == Z1, f"Number of ones changed {nZ1} vs. {Z1}"

    nZ6 = len(np.where(nZflat == 6)[0])
    Z6 = len(np.where(Zflat == 6)[0])
    assert nZ6 == Z6, f"Number of sixes changed {nZ6} vs. {Z6}"

    nZ7 = len(np.where(nZflat == 7)[0])
    Z7 = len(np.where(Zflat == 7)[0])
    assert nZ7 == Z7, f"Number of sevens changed {nZ7} vs. {Z7}"

    nZ8 = len(np.where(nZflat == 8)[0])
    Z8 = len(np.where(Zflat == 8)[0])
    assert nZ8 == Z8, f"Number of eights changed {nZ8} vs. {Z8}"




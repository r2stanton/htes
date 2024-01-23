from ase.io.trajectory import Trajectory
from ase import Atoms
import numpy as np


def pad_arrays(arrs, pad_value = 0):
    """
    Pad arrays according to the largest number of non-zero elements of any
    entry, or a specified size.
    """

def sort_by_element(arrs):
    """
    Sort a list of zero-padded array bys their element types. The first
    array passed must be the atomic numbers with shape (n_systems, n_max_atoms).

    All subsequent arrays will be sorted along the first axes as well, e.g.
    R -> (n_systems, n_max_atoms, 3) will be sorted along the n_max_atoms dim.

    Note:
    There is a much faster way to do this if all the arrays fit the same shape.
    """

    if type(arrs) != list:
        arrs = list(arrs)
    arrs = [arr.copy() for arr in arrs]

    n_mol = [arr.shape[0] for arr in arrs]
    assert len(set(n_mol)) == 1, "All arrays must have the same number of molecules."
    n_mol = n_mol[0]


    for i in range(n_mol):
        for arr_idx in range(len(arrs)):
            if arr_idx == 0:
                perm = np.argsort(arrs[0][i])[::-1]
                arrs[0][i] = arrs[0][i][perm]
            else:
                arrs[arr_idx][i] = arrs[arr_idx][i][perm]

    if len(arrs) == 1:
        return arrs[0]
    else:
        return (*arrs,)

def r_z_to_traj(r, z, output_name = "r_z.traj"):
    assert r.shape[0] == z.shape[0], "r and z must have the same number of molecules."
    n_mol = r.shape[0]

    traj = Trajectory(output_name, "w")

    for i in range(n_mol):
        atoms = Atoms(numbers = z[i], positions = r[i])
        traj.write(atoms)

def atomization_from_total_energy(E_tot, Z, E_iso, per_atom = False):
    """
    Computes and returns a numpy array of the same shape as E_tot which contains
    the atomization energies. E_tot should be a ndarray of total energies. E_iso
    should be a dictionary of Atomic Number:Isolated Energy. Z is species.
    """

    elements = list(E_iso.keys())

    E_at = np.zeros_like(E_tot)
    for i, z in enumerate(Z):
        res = 0
        for element in elements:
            n_ele = len(np.where(z == element)[0])
            res += n_ele*E_iso[element]
        if per_atom:
            E_at[i] = (E_tot[i] - res)/np.count_nonzero(z)
        else:
            E_at[i] = E_tot[i] - res

    return E_at



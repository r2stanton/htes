from ase.data import chemical_symbols
import matplotlib.pyplot as plt
from ase import Atoms
import numpy as np

def plot_atomwise_by_element(Q, Z, element_types, show = True, semilogy = False,
                             quantity_name = 'Charge |e|', colors = None):
    """
    Plot histograms of an atom-wise quantity (semi-empirical parameters, 
    chargest, etc.) as a histogram, by the element type. There will be one
    histogram per element.
    """
    if colors is not None:
        if type(colors) != str:
            if len(colors) != len(element_types):
                raise ValueError("Either provide no colors, one per element, \
                or a single color for all")

    Qf = Q.flatten()
    Zf = Z.flatten()

    n_el = len(element_types)
    near_sq = n_el

    while int(np.sqrt(near_sq)) < np.sqrt(near_sq):
        near_sq += 1

    gridsize = int(np.sqrt(near_sq))
    plt.subplots(gridsize, gridsize)
    mean_charges = {}
    for i in range(n_el):
        plt.subplot(gridsize, gridsize, i+1)

        curr_charges = Qf[np.where(Zf == element_types[i])[0]]

        if type(colors) == str:
            col = colors
        else:
            col = colors[i]

        plt.hist(curr_charges, rwidth = .85, color = col,
                 edgecolor = 'k', bins = 20)

        plt.xlabel(f"{chemical_symbols[element_types[i]]} {quantity_name}")
        if semilogy:
            plt.semilogy()

        q_mean = np.mean(curr_charges)

        plt.axvline(x=q_mean, linestyle = '--', color = 'orange',
                    label = f'Avg = {q_mean:.2f}')

        plt.legend()

        mean_charges[element_types[i]] = q_mean

    if show:
        plt.tight_layout()
        plt.show()

    return mean_charges

def plot_force_by_element(F, Z, element_types, show = True, semilogy = False,
                            quantity_name = 'Force |eV/A|  ', colors = None):
    """
    Plot histograms of a forces (or anything to be summed over atoms, e.g.
    atom features) by element. Basically a more specific case of 
    plot_atomwise_by_element above.
    """

    if colors is not None:
        if type(colors) != str:
            if len(colors) != len(element_types):
                raise ValueError("Either provide no colors, one per element, \
                or a single color for all")

    Fs = np.sqrt(np.einsum('...j, ...j', F, F))
    Ff = Fs.flatten()
    Zf = Z.flatten()

    n_el = len(element_types)
    near_sq = n_el

    while int(np.sqrt(near_sq)) < np.sqrt(near_sq):
        near_sq += 1

    gridsize = int(np.sqrt(near_sq))
    plt.subplots(gridsize, gridsize)
    mean_charges = {}
    for i in range(n_el):
        plt.subplot(gridsize, gridsize, i+1)

        curr_charges = Ff[np.where(Zf == element_types[i])[0]]

        if type(colors) == str:
            col = colors
        else:
            col = colors[i]
        plt.hist(curr_charges, rwidth = .85, color = col,
                 edgecolor = 'k', bins = 20)
        plt.xlabel(f"{chemical_symbols[element_types[i]]} {quantity_name}")
        if semilogy:
            plt.semilogy()

        q_mean = np.mean(curr_charges)

        plt.axvline(x=q_mean, linestyle = '--', color = 'orange',
                    label = f'Avg = {q_mean:.2f}')

        plt.legend()

        mean_charges[element_types[i]] = q_mean

    if show:
        plt.tight_layout()
        plt.show()

    return mean_charges


def plot_forces_by_nn(F, Z, R):
    """
    Plot force magnitudes by nearest neighbor distance. Currently hard coded to
    organic systems (C, H, N, O).
    """

    H_force_mags = []
    C_force_mags = []
    N_force_mags = []
    O_force_mags = []

    H_nn_dist = []
    C_nn_dist = []
    N_nn_dist = []
    O_nn_dist = []

    H_colors = []
    C_colors = []
    N_colors = []
    O_colors = []

    c_dict = {1: 'black', 6: 'grey', 7: 'blue', 8: 'red'}

    for idx, i in enumerate(F):
        n_at = np.count_nonzero(Z[idx])
        a = np.sqrt(np.einsum('...j, ...j', i, i))

        these_atoms = Atoms(numbers = Z[idx,:n_at], positions = R[idx, :n_at])
        dm = these_atoms.get_all_distances()
        dm += np.eye(dm.shape[0]) * 100

        if a.shape != Z[idx].shape:
            print(i)
            print(idx)
            print("ERROR: Shape mismatch")
            print(a)
            print(Z[idx])

        for jdx, j in enumerate(a):
            if Z[idx][jdx] == 1:
                H_force_mags.append(j)
                H_nn_dist.append(np.min(dm[jdx]))

                amin = np.argmin(dm[jdx])
                H_colors.append(c_dict[Z[idx][amin]])

            elif Z[idx][jdx] == 6:
                C_force_mags.append(j)
                C_nn_dist.append(np.min(dm[jdx]))

                amin = np.argmin(dm[jdx])
                C_colors.append(c_dict[Z[idx][amin]])

            elif Z[idx][jdx] == 7:
                N_force_mags.append(j)
                N_nn_dist.append(np.min(dm[jdx]))

                amin = np.argmin(dm[jdx])
                N_colors.append(c_dict[Z[idx][amin]])

            elif Z[idx][jdx] == 8:
                O_force_mags.append(j)
                O_nn_dist.append(np.min(dm[jdx]))

                amin = np.argmin(dm[jdx])
                O_colors.append(c_dict[Z[idx][amin]])

        if idx % 1000 == 0:
            print(idx)



    bins = np.arange(0, 16, .15)
    alpha = 1.0
    rw = 1.0
    color = 'blue'

    plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True,
                 figsize = (14, 8))
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

    plt.subplot(2, 2, 1)
    plt.title("BSE Singlets Force Magnitudes")
    plt.scatter(C_nn_dist, C_force_mags, s = .1, c = C_colors)
    plt.xlim(0.5, 2)
    plt.text(1, 50, "Carbon")
    plt.ylabel("$F_{mag}$ (eV/Angstrom)")
    # plt.xlabel("Force Magnitude (eV/Angstrom)")


    plt.subplot(2, 2, 2)
    plt.title("BSE Singlets Force Magnitudes")
    plt.scatter(O_nn_dist, O_force_mags, s = .1, c = O_colors)
    plt.text(1, 50, "Oxygen")
    plt.xlim(0.5, 2)
    # plt.xlabel("Force Magnitude (eV/Angstrom)")

    plt.subplot(2, 2, 3)
    plt.scatter(H_nn_dist, H_force_mags, s = .1, c = H_colors)
    plt.text(1, 50, "Hydrogen")
    plt.xlim(0.5, 2)
    plt.ylabel("$F_{mag}$ (eV/Angstrom)")
    plt.xlabel("Nearest Neighbor Distance (Angstrom)")
    # plt.xlabel("Force Magnitude (eV/Angstrom)")

    plt.subplot(2, 2, 4)
    plt.scatter(N_nn_dist, N_force_mags, s = .1, c = N_colors)
    plt.text(1, 50, "Nitrogen")
    plt.xlabel("Nearest Neighbor Distance (Angstrom)")
    plt.xlim(0.5, 2)
    # plt.xlabel("Force Magnitude (eV/Angstrom)")

    plt.show()


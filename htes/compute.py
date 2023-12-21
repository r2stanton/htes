import os 

"""
 The purpose of this file is for the computation of things which do not require
 extensive computational resources. E.g. some semiempirical or purely classical
 methods such as EMT from ASE, D3/H4 calculations, xTB via TBLite, etc.
"""


def compute_d3h4(Z, R, d3_path = None, h4_path = None):

    # If the paths to the executables are not passed to the function, 
    # then they must be set as the environment variables below.
    if d3_path is None:
        d3_path = os.getenv("D3_PATH")
    if h4_path is None:
        h4_path = os.getenv("H4_PATH")
    %%time
# element_dict = {1:  'H',
# 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', }
element_dict = {1:  'H', 6: 'C', 7: 'N', 8: 'O'}

log_file = open('log_D3H4.log', 'w+')


path_to_H4 = "/home/alg/ml/full_datset/d3h4/code/H4/h_bonds4"
path_to_D3 = "/home/alg/ml/full_datset/d3h4/code/D3/dftd3"

    E_D3_db = np.array([])
    E_H4_db = np.array([])

    force_dim = 20 # set force dim according to dimentions of data

    F_D3_db = np.empty(shape=(0, force_dim, 3))
    F_H4_db = np.empty(shape=(0, force_dim, 3))

    counter = 0
    for z, r in zip(Z, R):
        if counter%200 == 0:
            with open('log_D3H4.log', 'a') as log_file:
                log_file.write(str(counter)+'\n')
                
            with open('code/D3H4/E_D3_scan.npy', 'wb') as f:
                np.save(f, E_D3_db)
            with open('code/D3H4/E_H4_scan.npy', 'wb') as f:
                np.save(f, E_H4_db)
                
            with open('code/D3H4/F_D3_scan.npy', 'wb') as f:
                np.save(f, F_D3_db)
            with open('code/D3H4/F_H4_scan.npy', 'wb') as f:
                np.save(f, F_H4_db)
            print(counter)
            
        zero_pad_atoms = sum(z==0)
        xyz_lines = str(sum(z!=0)) + '\n\n'
        for atom, coord in zip(z, r):
            if atom != 0:
                xyz_lines += element_dict[atom] + ' ' + ' '.join(map("{:.5f}".format, coord)) + '\n'
                #print(element_dict[atom], ' '.join(map(str, coord)))
                
        #print(xyz_lines)
        # Compute d3/h4 corrections seperately
        E_H4, F_H4 = calc_H4(path_to_H4, xyz_lines)
        E_D3, F_D3 = calc_D3(path_to_D3, xyz_lines)
       
        # Corresponding unit conversions
        E_H4 = E_H4*0.043364115 # kcal/mol -> eV
        F_H4 = F_H4*0.043364115 # kcal/mol.A -> eV/A
        
        E_D3 = E_D3*0.043364115 # kcal/mol -> eV
        F_D3 = F_D3*51.42208619083232 # Ha/Bohr -> eV/A
        
        # Pad and stack
        zero_pad_F = np.repeat([[0,0,0]],zero_pad_atoms, axis=0)
        F_H4 = np.vstack((F_H4, zero_pad_F))
        F_D3 = np.vstack((F_D3, zero_pad_F))
        
        # Store in 'db' arrays.
        E_D3_db = np.append(E_D3_db, E_D3)
        E_H4_db = np.append(E_H4_db, E_H4)
        
        F_D3_db = np.append(F_D3_db, [F_D3], axis=0)
        F_H4_db = np.append(F_H4_db, [F_H4], axis=0)
        counter +=1

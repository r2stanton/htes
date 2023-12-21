from ase.data import chemical_symbols
import os, re, subprocess
import numpy as np

"""
 The purpose of this file is for the computation of things which do not require
 extensive computational resources. E.g. some semiempirical or purely classical
 methods such as EMT from ASE, D3/H4 calculations, xTB via TBLite, etc.
"""

def compute_d3h4(Z, R, path_to_D3 = None, path_to_H4 = None,
                 save_every = 1000):

    force_dim = Z.shape[1]
    # If the paths to the executables are not passed to the function, 
    # then they must be set as the environment variables below.
    if path_to_D3 is None:
        path_to_D3 = os.getenv("D3_PATH")
    if path_to_H4 is None:
        path_to_H4 = os.getenv("H4_PATH")


    log_file = open('log_D3H4.log', 'w+')

    E_D3_db = np.array([])
    E_H4_db = np.array([])

    F_D3_db = np.empty(shape=(0, force_dim, 3))
    F_H4_db = np.empty(shape=(0, force_dim, 3))

    counter = 0
    for z, r in zip(Z, R):
        if counter%save_every == 0:
            with open('log_D3H4.log', 'a') as log_file:
                log_file.write(str(counter)+'\n')
            with open('E_D3.npy', 'wb') as f:
                np.save(f, E_D3_db)
            with open('E_H4.npy', 'wb') as f:
                np.save(f, E_H4_db)
            with open('F_D3.npy', 'wb') as f:
                np.save(f, F_D3_db)
            with open('F_H4.npy', 'wb') as f:
                np.save(f, F_H4_db)
            print(counter)
            
        zero_pad_atoms = sum(z==0)
        xyz_lines = str(sum(z!=0)) + '\n\n'
        for atom, coord in zip(z, r):
            if atom != 0:
                xyz_lines += chemical_symbols[atom] + ' ' + ' '.join(map("{:.5f}".format, coord)) + '\n'
                
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

def calc_H4(path_to_H4, lines):
        
    result = subprocess.run( [ path_to_H4, '<' ], input=lines.encode(), capture_output=True )
    output = result.stdout.decode()
    regex_E = r'Total: \s*(\S+)\n'
    regex_F = r'Total gradient\n([\s\S]*)$'
    
    match_E = re.search(regex_E, output)
    match_F = re.search(regex_F, output)
    #print(output)
    
    if match_E is None or match_F is None:
        raise ValueError('Something is wrong with ')
    else:
        E = float(match_E.group(1).strip())
        F_string = match_F.group(1).strip()
        
        F_string = '\n'.join(line.strip() for line in F_string.strip().split('\n'))
        F = np.fromstring(F_string, sep=' ').reshape(-1, 3)
        return E, F

def calc_D3(path_to_D3, lines):
    
    tmp_f_name = 'tmp_D3.xyz'
    with open(tmp_f_name, 'w+') as f:
        f.write(lines)
        f.close
    
    result = subprocess.run([path_to_D3, tmp_f_name, '-func', 'pm6', '-zero', '-grad'], capture_output=True)
    output = result.stdout.decode()
    #print(output)
    regex_E = r'Edisp /kcal,au:\s*(\S+\s+)\S+\n'
    regex_F = r'Gradient \[au\]:\n([\s\S]*)normal '
    
    match_E = re.search(regex_E, output)    
    match_F = re.search(regex_F, output)
    if match_E is None:
        raise ValueError('Something is wrong with ')
    else:
        E = float(match_E.group(1).strip())
        F_string = match_F.group(1).strip()
        
        F_string = '\n'.join(line.strip() for line in F_string.strip().split('\n'))
        F = np.fromstring(F_string, sep=' ').reshape(-1, 3)
        
        return E, F

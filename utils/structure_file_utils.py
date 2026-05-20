import scipy.constants as sc
import mdtoolkit.logging as log

def estimate_number_density(density: float, molecular_weight : float, atom_count : int = 1):
    '''
    INPUT: \n
    density (float) : density of species in g/cm^3 \n
    molecular_weight (float) : molecular weight of species in g/mol \n
    atom_count (int) : number of atoms in species molecule. Default = 1 \n
    '''

    number_density = density/molecular_weight * sc.N_A/10**24 * atom_count

    return number_density

import itertools
from rdfpy import rdf
import numpy as np
from collections import defaultdict
from MDToolkit.data.objects import Simulation

def compute_type_wise_rdf(simulation : Simulation, rdf_cutoff = 10, bins = 250):
    atom_lists = [frame.get_atoms_list() for frame in simulation.frames]

    elements_set = set([atom.element for atom in atom_lists[0]]) # Assumes no atoms lost

    element_combinations = list(itertools.product(elements_set, repeat = 2))

    element_combinations = set(tuple(sorted(combination)) for combination in element_combinations)

    print(element_combinations)

def compute_total_rdf(simulation : Simulation, rdf_cutoff = 10, bins = 250):
    coordinate_lists = np.array([[atom.position for atom in frame.get_atoms_list()] for frame in simulation.frames])

    print(coordinate_lists[0].shape)

    total_rdfs = rdf(coordinate_lists[0], dr = rdf_cutoff / bins, rcutoff=rdf_cutoff)

    return total_rdfs
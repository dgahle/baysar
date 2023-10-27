# Imports
from numpy import array, sqrt, where

# Variables
elements = ["H", "D", "He", "Be", "B", "C", "N", "O", "Ne"]
masses = [1, 2, 4, 9, 10.8, 12, 14, 16, 20.2]
elements_and_masses = []
for elm, mass in zip(elements, masses):
    elements_and_masses.append((elm, mass))
atomic_masses = dict(elements_and_masses)


# Functions and classes
def species_to_element(species):
    return species[0 : where([a == "_" for a in species])[0][0]]


def get_atomic_mass(species):
    elm = species_to_element(species)
    return atomic_masses[elm]


if __name__ == "__main__":
    pass

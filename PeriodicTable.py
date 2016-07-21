# standard libraries
import fractions
import json
import os
import typing

# third party libraries
# None

# local libraries
# None

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


# see https://en.wikipedia.org/wiki/Electron_shell
# shell_number === principle quantum number n, or K, L, M, N, O, P, etc.
# subshell_index === EELS notation subshell
# K = 1s, L1 = 2s, L2 = 2p1/2, L3 = 2p3/2, M1 = 3s, M2 = 3p1/2, M3 = 3p3/2, M4 = 3d1/2, M5 = 3d3/2, etc.
class ElectronShell:
    def __init__(self, atomic_number: int, shell_number: int, subshell_index: int):
        self.atomic_number = atomic_number
        self.shell_number = shell_number
        self.subshell_index = subshell_index

    def __str__(self):
        return "{}-{}".format(PeriodicTable().element_symbol(self.atomic_number), self.get_shell_str_in_eels_notation(True))

    def to_long_str(self, include_subshell: bool=False):
        binding_energy_ev = PeriodicTable().nominal_binding_energy_ev(self)
        eels_shell_str = "{}-{}".format(PeriodicTable().element_symbol(self.atomic_number), self.get_shell_str_in_eels_notation(include_subshell))
        return "{}{}".format(eels_shell_str, " {:.1f} eV".format(binding_energy_ev) if binding_energy_ev is not None else str())

    def get_shell_str_in_eels_notation(self, include_subshell: bool=False) -> str:
        shell_str = chr(ord('K') + self.shell_number - 1)
        if (shell_str != 'K') and include_subshell:
            shell_str += str(self.subshell_index)
        return shell_str

    @classmethod
    def from_eels_notation(cls, atomic_number: int, eels_shell: str) -> "ElectronShell":
        shell_number = ord(eels_shell[0].upper()) - ord('K') + 1
        if eels_shell == "K":
            return ElectronShell(atomic_number, shell_number, 0)
        subshell_index = int(eels_shell[1:])
        return ElectronShell(atomic_number, shell_number, subshell_index)

    @property
    def azimuthal_quantum_number(self) -> int:
        aqn_table = (None, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        return aqn_table[self.subshell_index]

    @property
    def subshell_label(self) -> str:
        subshell_labels = ('a', 's', 'd', 'f', 'g', 'h', 'i', 'j')
        return subshell_labels[self.azimuthal_quantum_number]

    @property
    def spin_fraction(self) -> fractions.Fraction:
        spins = (None, 1, 1, 3, 3, 5, 5, 7, 7, 9)
        return fractions.Fraction(spins[self.azimuthal_quantum_number], 2)


class PeriodicTable(metaclass=Singleton):
    def __init__(self):
        dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        edge_data_file = os.path.join(dir, os.path.join("resources", "edges.json"))
        with open(edge_data_file, "r") as f:
            self.__edge_data = json.load(f)

    def element_symbol(self, atomic_number: int) -> str:
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == atomic_number:
                return edge_data_item.get("symbol")
        return None

    def nominal_binding_energy_ev(self, electron_shell: ElectronShell) -> float:
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == electron_shell.atomic_number:
                return edge_data_item.get("edges", dict()).get(electron_shell.get_shell_str_in_eels_notation(True))
        return None

    def get_elements_list(self) -> typing.Tuple[int, str]:
        """Return a list of tuples: atomic number, atomic symbol."""
        return ((edge_data_item.get("z"), edge_data_item.get("symbol")) for edge_data_item in self.__edge_data)

    def get_edges_list(self, atomic_number: int) -> typing.Tuple[int, str]:
        """Return a list of tuples: electron shell (lowest energy within shell number), edge name (without subshell)."""
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == atomic_number:
                edge_dict = edge_data_item.get("edges", dict())
                edge_map = dict()
                for eels_shell, energy in edge_dict.items():
                    electron_shell = ElectronShell.from_eels_notation(atomic_number, eels_shell)
                    base_electron_shell = edge_map.setdefault(electron_shell.shell_number, (None, 1E9))
                    if energy < base_electron_shell[1]:
                        edge_map[electron_shell.shell_number] = (electron_shell, energy)
                return list((edge_map[key][0], edge_map[key][0].to_long_str()) for key in sorted(edge_map.keys()))
        return None


# print(ElectronShell.from_eels_notation(6, "M4"))
# print(ElectronShell.from_eels_notation(6, "M4").get_shell_str_in_eels_notation(True))
# print(PeriodicTable().nominal_binding_energy_ev(ElectronShell.from_eels_notation(35, "M4")))

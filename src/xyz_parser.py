"""Strict parser for concatenated XYZ molecular geometry files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


_PERIODIC_TABLE = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh "
    "Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(_PERIODIC_TABLE)}


@dataclass(frozen=True)
class MolecularGeometry:
    """One XYZ frame with atom symbols, Cartesian coordinates, and source comment."""

    symbols: tuple[str, ...]
    coordinates: np.ndarray
    comment: str = ""

    def __post_init__(self) -> None:
        if len(self.symbols) == 0:
            raise ValueError("A molecular geometry must contain at least one atom.")
        if self.coordinates.shape != (len(self.symbols), 3):
            raise ValueError("Coordinates must have shape (n_atoms, 3).")
        unknown = sorted(set(self.symbols).difference(ATOMIC_NUMBERS))
        if unknown:
            raise ValueError(f"Unsupported chemical symbols: {unknown}")

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in this geometry."""
        return len(self.symbols)

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Return atomic numbers in current atom order."""
        return np.asarray([ATOMIC_NUMBERS[symbol] for symbol in self.symbols], dtype=float)


def read_concatenated_xyz(path: str | Path) -> list[MolecularGeometry]:
    """Parse all frames from a concatenated XYZ file.

    Blank lines between frames are allowed. Malformed atom counts, truncated frames,
    unknown elements, and non-numeric coordinates raise descriptive exceptions.
    """
    return list(iter_concatenated_xyz(path))


def iter_concatenated_xyz(path: str | Path) -> Iterator[MolecularGeometry]:
    """Yield one validated geometry at a time from a concatenated XYZ file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cursor = 0
    frame_index = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as error:
            raise ValueError(f"Frame {frame_index}: expected integer atom count at line {cursor + 1}.") from error
        if atom_count <= 0:
            raise ValueError(f"Frame {frame_index}: atom count must be positive.")
        cursor += 1
        if cursor >= len(lines):
            raise ValueError(f"Frame {frame_index}: missing comment line.")
        comment = lines[cursor]
        cursor += 1
        stop = cursor + atom_count
        if stop > len(lines):
            raise ValueError(f"Frame {frame_index}: expected {atom_count} atom lines but file ended early.")
        symbols: list[str] = []
        coordinates: list[list[float]] = []
        for atom_offset, line in enumerate(lines[cursor:stop]):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Frame {frame_index}, atom {atom_offset}: expected symbol and three coordinates.")
            symbol = fields[0]
            if symbol not in ATOMIC_NUMBERS:
                raise ValueError(f"Frame {frame_index}, atom {atom_offset}: unknown element symbol {symbol!r}.")
            try:
                xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
            except ValueError as error:
                raise ValueError(f"Frame {frame_index}, atom {atom_offset}: non-numeric coordinate.") from error
            symbols.append(symbol)
            coordinates.append(xyz)
        yield MolecularGeometry(tuple(symbols), np.asarray(coordinates, dtype=float), comment)
        cursor = stop
        frame_index += 1


def atom_count_summary(geometries: list[MolecularGeometry]) -> dict[str, int | float | dict[str, int]]:
    """Summarize atom-count distribution without imposing a maximum atom count."""
    counts = np.asarray([geometry.atom_count for geometry in geometries], dtype=int)
    if len(counts) == 0:
        return {"geometry_count": 0, "distribution": {}}
    unique, frequencies = np.unique(counts, return_counts=True)
    return {
        "geometry_count": int(len(counts)),
        "min_atoms": int(counts.min()),
        "max_atoms": int(counts.max()),
        "mean_atoms": float(counts.mean()),
        "distribution": {str(count): int(frequency) for count, frequency in zip(unique, frequencies)},
    }

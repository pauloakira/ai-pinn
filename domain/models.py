# Python libs
from dataclasses import dataclass

@dataclass
class TimoshenkoBeam:
    length: float
    G: float
    E: float
    As: float
    I: float

@dataclass
class TimoshenkoBoundary:
    w_0 : float
    w_L : float
    psi_0 : float
    psi_L : float
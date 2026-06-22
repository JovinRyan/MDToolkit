from MDToolkit.data.misc_objects import CylinderVolume
from MDToolkit.data.objects import Atom
import numpy as np

cyl = CylinderVolume(
    point1=[10, 10, 0],
    point2=[10, 10, 10],
    radius=3
)

cyl.discretize_axial()
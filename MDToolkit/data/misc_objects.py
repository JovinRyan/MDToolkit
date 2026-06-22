import numpy as np
from abc import ABC, abstractmethod
from MDToolkit.data.objects import Simulation, StructuredSystem, Molecule, Atom

class Volume(ABC):
    """
    Abstract base class for geometric volumes.
    """

    @property
    @abstractmethod
    def volume(self):
        pass

    @abstractmethod
    def contains(self, points):
        """
        Parameters
        ----------
        points : (N,3) ndarray

        Returns
        -------
        mask : (N,) bool ndarray
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self):
        """
        Returns
        -------
        mins, maxs : ndarray, ndarray
        """
        pass

class BoxVolume(Volume):

    def __init__(self, point1, point2):

        self.point1 = np.asarray(point1, dtype=float)
        self.point2 = np.asarray(point2, dtype=float)

        self.mins = np.minimum(self.point1, self.point2)
        self.maxs = np.maximum(self.point1, self.point2)
    
    def __repr__(self):
        return f"BoxVolume(bounding_box={self.bounding_box}, volume={self.volume})"

    @property
    def volume(self):

        lengths = self.maxs - self.mins
        return np.prod(lengths)

    @property
    def bounding_box(self):

        return self.mins, self.maxs

    def contains(self, points):

        points = np.asarray(points)

        return np.all(
            (points >= self.mins) &
            (points <= self.maxs),
            axis=1
        )
    
    def contains_atoms(self, atoms):

        points = np.asarray([atom.position for atom in atoms])

        return np.all(
            (points >= self.mins) &
            (points <= self.maxs),
            axis=1
        )
    
    def discretize_axial(self, n_bins=250, axis="x"):

        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2
        }

        axis = axis.lower()

        if axis not in axis_map:
            raise ValueError(
                f"Axis '{axis}' must be x, y, or z"
            )

        axis_idx = axis_map[axis]

        edges = np.linspace(
            self.mins[axis_idx],
            self.maxs[axis_idx],
            n_bins + 1
        )

        bins = []

        for i in range(n_bins):

            p1 = self.mins.copy()
            p2 = self.maxs.copy()

            p1[axis_idx] = edges[i]
            p2[axis_idx] = edges[i + 1]

            bins.append(BoxVolume(p1, p2))

        return bins


class CylinderVolume(Volume):

    def __init__(
        self,
        point1,
        point2,
        radius
    ):

        self.point1 = np.asarray(point1, dtype=float)
        self.point2 = np.asarray(point2, dtype=float)

        self.radius = float(radius)

        axis = self.point2 - self.point1

        self.height = np.linalg.norm(axis)

        if self.height == 0:
            raise ValueError(
                "Cylinder endpoints cannot coincide."
            )

        self.axis = axis / self.height

        self.axis_idx = np.argmax(np.abs(self.axis))

    @property
    def volume(self):

        return np.pi * self.radius**2 * self.height

    @property
    def bounding_box(self):

        mins = np.minimum(
            self.point1,
            self.point2
        ) - self.radius

        maxs = np.maximum(
            self.point1,
            self.point2
        ) + self.radius

        return mins, maxs
    
    def __repr__(self):
        return (
            f"CylinderVolume("
            f"point1={self.point1.tolist()}, "
            f"point2={self.point2.tolist()}, "
            f"radius={self.radius:.4f}, "
            f"volume={self.volume:.4f})"
        )

    def contains(self, points):

        points = np.asarray(points)

        print(points)

        rel = points - self.point1

        axial = np.dot(rel, self.axis)

        radial_vec = (
            rel -
            np.outer(axial, self.axis)
        )

        radial_dist = np.linalg.norm(
            radial_vec,
            axis=1
        )

        return (
            (axial >= 0.0) &
            (axial <= self.height) &
            (radial_dist <= self.radius)
        )
    
    def contains_atoms(self, atoms):
        
        coordinates = np.asanyarray([atom.position for atom in atoms])

        print(coordinates)

        rel = coordinates - self.point1

        axial = np.dot(rel, self.axis)

        radial_vec = (
            rel -
            np.outer(axial, self.axis)
        )

        radial_dist = np.linalg.norm(
            radial_vec,
            axis=1
        )

        return (
            (axial >= 0.0) &
            (axial <= self.height) &
            (radial_dist <= self.radius)
        )
    
    def discretize_axial(self, n_bins = 250):
        '''
        '''
        axes_bounds = (self.point1[self.axis_idx], self.point2[self.axis_idx])
        points = np.linspace(min(axes_bounds), max(axes_bounds), n_bins + 1)

        bins = []

        for i in range(1, len(points)):
            p1 = self.point1.copy()
            p2 = self.point2.copy()

            p1[self.axis_idx] = points[i - 1]
            p2[self.axis_idx] = points[i] 

            bins.append(CylinderVolume(p1, p2, self.radius))
        
        print(bins)
        return bins
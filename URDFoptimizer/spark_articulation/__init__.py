"""SPARK-faithful articulation-only optimization.

This package intentionally excludes SPARK mesh generation components. It works
with existing segmented meshes referenced by a URDF and optimizes continuous
articulation parameters against an open-state silhouette.
"""

from .urdf_model import ArticulatedURDF, JointSpec, LinkSpec, VisualSpec, parse_urdf
from .fk import compute_link_transforms, compute_visual_transform
from .model import DifferentiableArticulationModel, MeshPartTensors

__all__ = [
    "ArticulatedURDF",
    "JointSpec",
    "LinkSpec",
    "VisualSpec",
    "parse_urdf",
    "compute_link_transforms",
    "compute_visual_transform",
    "DifferentiableArticulationModel",
    "MeshPartTensors",
]


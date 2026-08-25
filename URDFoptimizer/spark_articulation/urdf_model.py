"""URDF parsing and writeback for articulation-only SPARK refinement."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]


def _parse_vec3(text: Optional[str], default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if text is None:
        return default
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected three floats, got: {text}")
    return vals[0], vals[1], vals[2]


def _parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    return float(text)


def _format_vec3(values: Sequence[float]) -> str:
    return f"{values[0]:.8f} {values[1]:.8f} {values[2]:.8f}"


@dataclass(frozen=True)
class VisualSpec:
    mesh_filename: Optional[str]
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    scale: Optional[Vec3] = None


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: List[VisualSpec] = field(default_factory=list)


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 0.0)
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None

    @property
    def is_revolute(self) -> bool:
        return self.joint_type == "revolute"

    @property
    def is_prismatic(self) -> bool:
        return self.joint_type == "prismatic"

    @property
    def is_fixed(self) -> bool:
        return self.joint_type == "fixed"


@dataclass
class ArticulatedURDF:
    robot_name: str
    links: Dict[str, LinkSpec]
    joints: List[JointSpec]
    source_path: Optional[Path] = None

    @property
    def joints_by_name(self) -> Dict[str, JointSpec]:
        return {joint.name: joint for joint in self.joints}

    @property
    def joints_by_child(self) -> Dict[str, JointSpec]:
        return {joint.child: joint for joint in self.joints}

    @property
    def children_by_parent(self) -> Dict[str, List[JointSpec]]:
        children: Dict[str, List[JointSpec]] = {}
        for joint in self.joints:
            children.setdefault(joint.parent, []).append(joint)
        return children

    @property
    def root_links(self) -> List[str]:
        child_links = {joint.child for joint in self.joints}
        return [name for name in self.links if name not in child_links]

    def topological_joints(self) -> List[JointSpec]:
        """Return joints ordered parent-before-child."""
        roots = self.root_links
        if not roots:
            raise ValueError("URDF has no root link; every link appears as a child")

        by_parent = self.children_by_parent
        ordered: List[JointSpec] = []
        queue = list(roots)
        visited_links = set(queue)

        while queue:
            parent = queue.pop(0)
            for joint in by_parent.get(parent, []):
                ordered.append(joint)
                if joint.child not in visited_links:
                    visited_links.add(joint.child)
                    queue.append(joint.child)

        if len(ordered) != len(self.joints):
            missing = {joint.name for joint in self.joints} - {joint.name for joint in ordered}
            raise ValueError(f"URDF joint graph is disconnected or cyclic; missing joints: {sorted(missing)}")
        return ordered


def _parse_visual(visual_el: ET.Element) -> Optional[VisualSpec]:
    origin = visual_el.find("origin")
    xyz = _parse_vec3(origin.attrib.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
    rpy = _parse_vec3(origin.attrib.get("rpy")) if origin is not None else (0.0, 0.0, 0.0)

    mesh = visual_el.find("./geometry/mesh")
    if mesh is None:
        return None

    scale_text = mesh.attrib.get("scale")
    scale = _parse_vec3(scale_text) if scale_text is not None else None
    return VisualSpec(mesh_filename=mesh.attrib.get("filename"), xyz=xyz, rpy=rpy, scale=scale)


def parse_urdf(path: str | os.PathLike[str]) -> ArticulatedURDF:
    urdf_path = Path(path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: Dict[str, LinkSpec] = {}
    for link_el in root.findall("link"):
        name = link_el.attrib["name"]
        visuals = []
        for visual_el in link_el.findall("visual"):
            visual = _parse_visual(visual_el)
            if visual is not None:
                visuals.append(visual)
        links[name] = LinkSpec(name=name, visuals=visuals)

    joints: List[JointSpec] = []
    for joint_el in root.findall("joint"):
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            raise ValueError(f"Joint {joint_el.attrib.get('name', '<unnamed>')} is missing parent or child")

        origin = joint_el.find("origin")
        axis = joint_el.find("axis")
        limit = joint_el.find("limit")
        joints.append(
            JointSpec(
                name=joint_el.attrib.get("name", f"{parent_el.attrib['link']}_to_{child_el.attrib['link']}"),
                joint_type=joint_el.attrib.get("type", "fixed"),
                parent=parent_el.attrib["link"],
                child=child_el.attrib["link"],
                xyz=_parse_vec3(origin.attrib.get("xyz")) if origin is not None else (0.0, 0.0, 0.0),
                rpy=_parse_vec3(origin.attrib.get("rpy")) if origin is not None else (0.0, 0.0, 0.0),
                axis=_parse_vec3(axis.attrib.get("xyz")) if axis is not None else (0.0, 0.0, 0.0),
                limit_lower=_parse_float(limit.attrib.get("lower")) if limit is not None else None,
                limit_upper=_parse_float(limit.attrib.get("upper")) if limit is not None else None,
            )
        )

    return ArticulatedURDF(
        robot_name=root.attrib.get("name", "object"),
        links=links,
        joints=joints,
        source_path=urdf_path,
    )


def resolve_mesh_path(mesh_path: Optional[str], urdf_path: str | os.PathLike[str]) -> Optional[Path]:
    if mesh_path is None:
        return None
    normalized = mesh_path
    if normalized.startswith("package://"):
        normalized = normalized.replace("package://", "", 1)
    path = Path(normalized)
    if path.is_absolute():
        return path
    return Path(urdf_path).resolve().parent / path


def _origin_xyz(origin: Optional[ET.Element]) -> Vec3:
    if origin is None:
        return 0.0, 0.0, 0.0
    return _parse_vec3(origin.attrib.get("xyz"))


def _ensure_origin(parent_el: ET.Element) -> ET.Element:
    origin = parent_el.find("origin")
    if origin is None:
        origin = ET.SubElement(parent_el, "origin")
        origin.attrib["xyz"] = "0 0 0"
        origin.attrib["rpy"] = "0 0 0"
    return origin


def _rpy_rotation_transpose_times_delta(rpy: Vec3, delta: Sequence[float]) -> Vec3:
    # This is R(rpy)^T @ delta, kept local to avoid a hard numpy dependency.
    import math

    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return (
        R[0][0] * delta[0] + R[1][0] * delta[1] + R[2][0] * delta[2],
        R[0][1] * delta[0] + R[1][1] * delta[1] + R[2][1] * delta[2],
        R[0][2] * delta[0] + R[1][2] * delta[1] + R[2][2] * delta[2],
    )


def _add_vec3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub_vec3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _iter_child_local_origins(root: ET.Element, child_link: str) -> Iterable[ET.Element]:
    """Origins expressed in the child link frame that must be compensated."""
    link = root.find(f"./link[@name='{child_link}']")
    if link is not None:
        for visual in link.findall("visual"):
            yield _ensure_origin(visual)
        for collision in link.findall("collision"):
            yield _ensure_origin(collision)

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.attrib.get("link") == child_link:
            yield _ensure_origin(joint)


def write_refined_urdf(
    input_urdf: str | os.PathLike[str],
    output_urdf: str | os.PathLike[str],
    origin_deltas: Dict[str, Sequence[float]],
    *,
    preserve_zero_pose: bool = True,
) -> None:
    """Write a URDF with refined joint origins.

    The mesh vertices are never changed. When ``preserve_zero_pose`` is true,
    child-local visual, collision, and outgoing joint origins are compensated so
    the existing assembled mesh remains unchanged at q=0 while the moving part
    rotates about the refined pivot for q!=0.
    """
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    for joint_name, delta in origin_deltas.items():
        if len(delta) != 3:
            raise ValueError(f"Origin delta for {joint_name} must have length 3")

        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Joint not found in URDF: {joint_name}")

        origin = _ensure_origin(joint)
        old_xyz = _origin_xyz(origin)
        rpy = _parse_vec3(origin.attrib.get("rpy"))
        new_xyz = _add_vec3(old_xyz, delta)
        origin.attrib["xyz"] = _format_vec3(new_xyz)

        child = joint.find("child")
        if preserve_zero_pose and child is not None:
            child_link = child.attrib["link"]
            delta_in_joint_frame = _rpy_rotation_transpose_times_delta(rpy, delta)
            compensation = (-delta_in_joint_frame[0], -delta_in_joint_frame[1], -delta_in_joint_frame[2])
            for child_origin in _iter_child_local_origins(root, child_link):
                cur_xyz = _origin_xyz(child_origin)
                child_origin.attrib["xyz"] = _format_vec3(_add_vec3(cur_xyz, compensation))

    ET.indent(tree, space="  ", level=0)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)


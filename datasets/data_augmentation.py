#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render URDF at different joint angles using the same camera/lighting as datasets/preprocess/render.py

This script:
1. Parses URDF to extract parent and child links
2. Loads meshes from GLB files referenced in URDF (already split by split_glb.py)
3. Applies forward kinematics to position child mesh at specified joint angles
4. Renders using pyrender with same settings as render.py (RADIUS=4, FOV=40, etc.)
5. Outputs rendered images for each specified angle

Usage:
    python render_urdf_angles.py --urdf render/mobility.urdf --angles 0,120,160 --output render/output/
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List
from pathlib import Path
import shutil
import json

import numpy as np
from PIL import Image
import trimesh

# Camera and rendering settings (matching datasets/preprocess/render.py)
RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 3.0
NUM_ENV_LIGHTS = 36


# --------------------------
# URDF parsing utilities
# --------------------------

def _parse_xyz(s: Optional[str]) -> np.ndarray:
    if s is None:
        return np.zeros(3, dtype=np.float32)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"xyz must have 3 values, got: {s}")
    return np.array(vals, dtype=np.float32)


def _parse_rpy(s: Optional[str]) -> np.ndarray:
    if s is None:
        return np.zeros(3, dtype=np.float32)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"rpy must have 3 values, got: {s}")
    return np.array(vals, dtype=np.float32)


def rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Convert roll-pitch-yaw to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=np.float32)
    return R


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for rotation matrix from axis-angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s,   x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,   c + z*z*C]
    ], dtype=np.float32)
    return R


def make_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 SE3 matrix from 3x3 rotation and 3D translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def find_single_revolute_joint(root: ET.Element, joint_name: Optional[str] = None) -> ET.Element:
    joints = []
    for j in root.findall('joint'):
        if j.attrib.get('type') == 'revolute':
            joints.append(j)
    if joint_name:
        for j in joints:
            if j.attrib.get('name') == joint_name:
                return j
        raise ValueError(f"Revolute joint named '{joint_name}' not found.")
    if len(joints) == 0:
        raise ValueError("No revolute joint found in URDF.")
    if len(joints) > 1:
        print("[WARN] Multiple revolute joints found; using the first. Use --joint-name to specify.")
    return joints[0]


def find_all_revolute_joints(root: ET.Element) -> list:
    """Find all revolute joints in URDF."""
    joints = []
    for j in root.findall('joint'):
        if j.attrib.get('type') == 'revolute':
            joints.append(j)
    return joints


def get_joint_max_angle(joint: ET.Element) -> float:
    """Get the upper limit angle for a revolute joint in radians."""
    limit = joint.find('limit')
    if limit is not None and 'upper' in limit.attrib:
        return float(limit.attrib['upper'])
    # Default max angle if no limit specified (180 degrees)
    return math.pi


def get_link_first_visual(root: ET.Element, link_name: str) -> Tuple[Optional[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns (mesh_path, origin_xyz, origin_rpy, scale) for the first <visual> of the link.
    """
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link '{link_name}' not found.")
    vis = link.find('visual')
    if vis is None:
        print(f"[WARN] Link '{link_name}' has no <visual>.")
        return None, np.zeros(3, np.float32), np.zeros(3, np.float32), None
    origin = vis.find('origin')
    xyz = _parse_xyz(origin.attrib.get('xyz')) if origin is not None else np.zeros(3, np.float32)
    rpy = _parse_rpy(origin.attrib.get('rpy')) if origin is not None else np.zeros(3, np.float32)
    geom = vis.find('geometry')
    if geom is None:
        print(f"[WARN] Link '{link_name}' visual has no <geometry>.")
        return None, xyz, rpy, None
    mesh = geom.find('mesh')
    if mesh is None:
        print(f"[WARN] Link '{link_name}' visual geometry is not a <mesh>.")
        return None, xyz, rpy, None
    filename = mesh.attrib.get('filename')
    scale = mesh.attrib.get('scale')
    scale = np.array([float(v) for v in scale.split()]) if scale is not None else None
    return filename, xyz, rpy, scale

def get_link_visuals(root: ET.Element, link_name: str) -> List[Tuple[Optional[str], np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Return all <visual> elements for the given link, in the order they appear in the URDF.
    Each element is (mesh_path, origin_xyz, origin_rpy, scale).
    Only <geometry><mesh> entries are returned; other types (e.g. box/cylinder) are skipped.
    """
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link '{link_name}' not found.")

    visuals = link.findall('visual')
    results: List[Tuple[Optional[str], np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

    for vis in visuals:
        origin = vis.find('origin')
        xyz = _parse_xyz(origin.attrib.get('xyz')) if origin is not None else np.zeros(3, np.float32)
        rpy = _parse_rpy(origin.attrib.get('rpy')) if origin is not None else np.zeros(3, np.float32)

        geom = vis.find('geometry')
        if geom is None:
            continue
        mesh = geom.find('mesh')
        if mesh is None:
            # Not a mesh (possibly box/cylinder), skip
            continue

        filename = mesh.attrib.get('filename')
        scale = mesh.attrib.get('scale')
        scale = np.array([float(v) for v in scale.split()]) if scale is not None else None

        results.append((filename, xyz, rpy, scale))

    if not results:
        # No mesh visual found; return a placeholder (consistent with original get_link_first_visual behavior)
        print(f"[WARN] Link '{link_name}' has no <visual>/<mesh>.")
        results.append((None, np.zeros(3, np.float32), np.zeros(3, np.float32), None))

    return results

def resolve_mesh_path(mesh_path: str, urdf_dir: str) -> str:
    if mesh_path is None:
        return None
    # Handle package:// or relative paths
    if mesh_path.startswith("package://"):
        mesh_path = mesh_path.replace("package://", "")
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(urdf_dir, mesh_path)
    return mesh_path


# --------------------------
# Mesh loading and FK
# --------------------------

def load_mesh(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float = 1.0) -> trimesh.Trimesh:
    """Load mesh from file and apply scaling."""
    if mesh_path is None or not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    tm = trimesh.load(mesh_path, force='mesh')

    if tm.is_empty:
        raise ValueError(f"Mesh is empty: {mesh_path}")

    # Apply URDF scale
    if scale is not None:
        if np.isscalar(scale):
            tm.apply_scale(float(scale))
        else:
            # Anisotropic scaling
            tm.vertices = tm.vertices * scale

    # Apply unit scale
    if unit_scale != 1.0:
        tm.apply_scale(float(unit_scale))

    return tm


def apply_forward_kinematics(
    parent_mesh: trimesh.Trimesh,
    child_mesh: trimesh.Trimesh,
    parent_vis_xyz: np.ndarray,
    parent_vis_rpy: np.ndarray,
    joint_xyz: np.ndarray,
    joint_rpy: np.ndarray,
    joint_axis: np.ndarray,
    joint_angle: float,
    child_vis_xyz: np.ndarray,
    child_vis_rpy: np.ndarray
) -> trimesh.Scene:
    """
    Apply forward kinematics to create a scene with parent and child at the specified joint angle.

    Transform chain:
    - Parent: T_parent = T_parent_visual
    - Child: T_child = T_parent_visual @ T_joint @ T_rotation @ T_child_visual
    """
    scene = trimesh.Scene()

    # Parent transform
    Rp = rpy_to_matrix(tuple(parent_vis_rpy))
    Tp_parent = make_SE3(Rp, parent_vis_xyz)

    # Apply parent transform
    parent_transformed = parent_mesh.copy()
    parent_transformed.apply_transform(Tp_parent)
    scene.add_geometry(parent_transformed, node_name='parent')

    # Child transform chain
    # 1. Joint origin
    Rj = rpy_to_matrix(tuple(joint_rpy))
    Tj = make_SE3(Rj, joint_xyz)

    # 2. Joint rotation (around axis by angle)
    Rrot = axis_angle_to_matrix(joint_axis, joint_angle)
    Trot = make_SE3(Rrot, np.zeros(3, dtype=np.float32))

    # 3. Child visual origin
    Rc = rpy_to_matrix(tuple(child_vis_rpy))
    Tc = make_SE3(Rc, child_vis_xyz)

    # Combined transform: world = Tp @ Tj @ Trot @ Tc
    T_child_world = Tp_parent @ Tj @ Trot @ Tc

    # Apply child transform
    child_transformed = child_mesh.copy()
    child_transformed.apply_transform(T_child_world)
    scene.add_geometry(child_transformed, node_name='child')

    return scene

def apply_multi_joint_fk(root: ET.Element, urdf_dir: str, joint_angles: dict, unit_scale: float = 1.0) -> trimesh.Scene:
    “””
    Visual-space FK:
    For each link, maintain its visual transform in world space directly:
      T_visual_world[root] = Tv(root)
      T_visual_world[child] = T_visual_world[parent] @ Tj @ Trot @ Tv(child)
    This matches the single-joint script's composition
      Tp_parent @ Tj @ Trot @ Tc_child
    at any depth.
    “””
    scene = trimesh.Scene()

    # ---- Pre-processing: collect link set, joint list, root link ----
    all_links = {link.attrib['name'] for link in root.findall('link')}
    child_links = {j.find('child').attrib['link'] for j in root.findall('joint') if j.find('child') is not None}
    root_links = all_links - child_links
    joints = root.findall('joint')

    if not root_links:
        raise ValueError("No root link found (all links are children)")

    print(f"[INFO] Found {len(all_links)} links, {len(joints)} joints")
    print(f"[INFO] Root links: {root_links}")

    # ---- Select reference link for alignment (more robust) ----
    # 1) Prefer the child of a fixed joint whose parent is 'base' (usually the main body)
    ref_link = None
    for _j in joints:
        if _j.attrib.get('type') == 'fixed':
            p = _j.find('parent').attrib['link']
            c = _j.find('child').attrib['link']
            if p == 'base':
                ref_link = c
                break

    # 2) Fallback to the parent link of the first revolute joint
    if ref_link is None:
        for _j in joints:
            if _j.attrib.get('type') == 'revolute':
                ref_link = _j.find('parent').attrib['link']
                break

    if ref_link is None:
        raise ValueError("No suitable reference link found for alignment (need a fixed-from-base child or a revolute parent).")

    print(f"[INFO] Reference base link for alignment: {ref_link}")


    # ---- 1) Pre-compute visual origin Tv(link) for each link ----
    Tv_map = {}
    for link in root.findall('link'):
        name = link.attrib['name']
        visuals = get_link_visuals(root, name)  # all <visual> elements
        # Prefer GLB as the reference visual for this link; fall back to the first one
        preferred = None
        for (mesh_path, vxyz, vrpy, scale) in visuals:
            fn = (mesh_path or "").lower()
            if fn.endswith(".glb") or "glb/" in fn:
                preferred = (mesh_path, vxyz, vrpy, scale)
                break
        if preferred is None:
            preferred = visuals[0]
        _, vxyz, vrpy, _ = preferred
        Rv = rpy_to_matrix(tuple(vrpy))
        Tv_map[name] = make_SE3(Rv, vxyz)


    # ---- 2) Maintain world-space visual transform map ----
    # Note: link-frame transforms are not maintained separately; visual world transforms are tracked
    # directly to match the single-joint script's composition order (parent visual first).
    T_visual_world = {}
    T_link_world   = {}  # link-frame world transforms

    for root_link in root_links:
        T_visual_world[root_link] = Tv_map[root_link].copy()
        T_link_world[root_link]   = np.eye(4, dtype=np.float32)  # because T_visual = T_link @ Tv_map
        print(f"[DEBUG] Root visual world for '{root_link}' initialized")


    # Iterate over all joints in topological order (process child only after parent is ready)
    processed = set()
    max_iterations = len(joints) + 2
    for _ in range(max_iterations):
        progressed = False
        for joint in joints:
            jname = joint.attrib.get('name', 'unnamed')
            if jname in processed:
                continue

            parent_link = joint.find('parent').attrib['link']
            child_link  = joint.find('child').attrib['link']
            if parent_link not in T_visual_world:
                continue  # parent not ready yet, defer to next iteration

            # joint origin
            jorigin = joint.find('origin')
            j_xyz = _parse_xyz(jorigin.attrib.get('xyz') if jorigin is not None else None)
            j_rpy = _parse_rpy(jorigin.attrib.get('rpy') if jorigin is not None else None)
            Rj = rpy_to_matrix(tuple(j_rpy))
            Tj = make_SE3(Rj, j_xyz)

            # joint rotation
            if joint.attrib.get('type') == 'revolute':
                jaxis_el = joint.find('axis')
                if jaxis_el is not None and 'xyz' in jaxis_el.attrib:
                    jaxis = _parse_xyz(jaxis_el.attrib['xyz'])
                    angle = float(joint_angles.get(jname, 0.0))
                    Rrot = axis_angle_to_matrix(jaxis, angle)
                    Trot = make_SE3(Rrot, np.zeros(3, dtype=np.float32))
                else:
                    Trot = np.eye(4, dtype=np.float32)
            else:
                Trot = np.eye(4, dtype=np.float32)

            # --- Key: propagate from parent visual world transform to child visual world transform ---
            T_visual_world[child_link] = T_visual_world[parent_link] @ Tj @ Trot @ Tv_map[child_link]
            T_link_world[child_link]   = T_visual_world[parent_link] @ Tj @ Trot

            print(f"[DEBUG] Joint '{jname}': {parent_link} -> {child_link}")
            print(f"[DEBUG]   Joint origin xyz: {j_xyz}, rpy: {j_rpy}, angle(deg)={np.rad2deg(joint_angles.get(jname, 0.0)):.2f}")
            processed.add(jname)
            progressed = True

        if not progressed:
            break

    if len(processed) < len(joints):
        print(f"[WARN] Could not process all joints ({len(processed)}/{len(joints)})")

    # ---- Key: rebase to reference link's visual transform to match the single-joint script ----
    # Goal: make the reference link's world transform equal Tv_map[ref_link] (same as single-joint),
    #       and left-multiply all other links by the same S to preserve relative poses.
    if ref_link not in T_visual_world:
        raise ValueError(f"Reference link '{ref_link}' transform not ready")
    Tv_ref = Tv_map[ref_link]
    # Current world transform of the reference link
    T_ref_cur = T_visual_world[ref_link]
    # Compute left-multiply alignment matrix S such that S @ T_ref_cur = Tv_ref
    S = Tv_ref @ np.linalg.inv(T_ref_cur)

    # Apply to all links
    for k in list(T_visual_world.keys()):
        T_visual_world[k] = S @ T_visual_world[k]
        T_link_world[k]   = S @ T_link_world[k]

    # ---- 3) Load meshes and apply visual world transforms ----
    for link in root.findall('link'):
        link_name = link.attrib['name']

        if link_name not in T_link_world:
            print(f"[WARN] No link-frame transform for link: {link_name}, skipping")
            continue

        visuals = get_link_visuals(root, link_name)
        if not visuals:
            continue

        for vi, (mesh_path, vis_xyz, vis_rpy, scale) in enumerate(visuals):
            if mesh_path is None:
                continue

            resolved = resolve_mesh_path(mesh_path, urdf_dir)
            if not os.path.exists(resolved):
                print(f"[WARN] Mesh not found: {resolved}")
                continue

            mesh = load_mesh(resolved, scale, unit_scale)

            # This component's own visual transform
            Rv = rpy_to_matrix(tuple(vis_rpy))
            Tv = make_SE3(Rv, vis_xyz)

            # Correct world transform: link-frame world transform @ this component's visual transform
            T_world = T_link_world[link_name] @ Tv

            print(f"[DEBUG] Link '{link_name}' visual[{vi}]: {os.path.basename(resolved)}")
            print(f"[DEBUG]   vis xyz: {vis_xyz}, rpy: {vis_rpy}")
            print(f"[DEBUG]   world T[:3,3]: {T_world[:3, 3]}")

            mesh_tf = mesh.copy()
            mesh_tf.apply_transform(T_world)

            node_name = f"{link_name}__v{vi}__{os.path.basename(resolved)}"
            scene.add_geometry(mesh_tf, node_name=node_name)

    return scene


# --------------------------
# Rendering (using src.utils.render_utils)
# --------------------------

def render_scene(scene: trimesh.Scene, output_path: str,
                 translation: np.ndarray = None, scale: float = None,
                 azimuth: float = 0.0, elevation: float = 0.0):
    """
    Render scene using the same settings as datasets/preprocess/render.py

    Settings:
    - RADIUS = 4
    - IMAGE_SIZE = (2048, 2048)
    - LIGHT_INTENSITY = 2.0
    - NUM_ENV_LIGHTS = 36

    Args:
        scene: Trimesh scene to render
        output_path: Output image path
        translation: Fixed translation for normalization (if None, compute from scene)
        scale: Fixed scale factor for normalization (if None, compute from scene)
        azimuth: Camera azimuth angle in degrees (rotation around Y axis)
        elevation: Camera elevation angle in degrees (rotation from XZ plane)

    Camera coordinate system:
        - azimuth=0, elevation=0: camera at (0, 0, radius) looking at origin along -Z
        - azimuth rotates around Y axis (0=front/+Z, 90=right/+X, 180=back/-Z, 270=left/-X)
        - elevation rotates from XZ plane (0=horizontal, 90=top/+Y, -90=bottom/-Y)

    Returns:
        (translation, scale) tuple used for normalization
    """
    from src.utils.render_utils import render_single_view

    # Copy scene to avoid modifying original
    scene_copy = scene.copy()

    if translation is None or scale is None:
        # Compute normalization parameters using the SAME method as normalize_mesh
        # in src.utils.data_utils (lines 14-18)
        bbox = scene_copy.bounding_box
        computed_translation = -bbox.centroid
        computed_scale = 2.0 / bbox.primitive.extents.max()

        if translation is None:
            translation = computed_translation
        if scale is None:
            scale = computed_scale

    # Apply the same normalization as normalize_mesh
    scene_copy.apply_translation(translation)
    scene_copy.apply_scale(scale)

    # Convert to geometry for rendering
    geometry = scene_copy.to_geometry()

    # Render single view with camera at specified azimuth/elevation
    image = render_single_view(
        geometry,
        azimuth=azimuth,
        elevation=elevation,
        radius=RADIUS,    # Fixed distance
        image_size=IMAGE_SIZE,
        fov=40.0,         # Fixed FOV
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil'
    )

    # Save image
    image.save(output_path)
    print(f"Saved rendering to: {output_path}")

    return translation, scale


# --------------------------
# Main
# --------------------------

def update_object_part_configs(config_path: Path, new_entries: list):
    """Update object_part_configs.json with new entries."""
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            configs = json.load(f)
    else:
        configs = []

    # Add new entries
    configs.extend(new_entries)

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)

    print(f"Updated {config_path} with {len(new_entries)} new entries")


def process_single_object(input_subfolder: Path, output_base: Path,
                         source_config_path: Path = None,
                         unit_scale: float = 1.0,
                         azimuth: float = 0.0, elevation: float = 0.0):
    """Process a single object: render at max and mid joint angles and copy to output.

    Args:
        input_subfolder: Input subfolder containing URDF
        output_base: Output base directory
        source_config_path: Path to source config (for looking up num_parts)
        unit_scale: Scale factor for meshes
        azimuth: Camera azimuth angle
        elevation: Camera elevation angle

    Returns:
        list: List of new config entries created (for updating object_part_configs.json)
    """
    subfolder_name = input_subfolder.name
    print(f"\n{'='*80}")
    print(f"Processing: {subfolder_name}")
    print(f"{'='*80}")

    # Find URDF file
    urdf_path = input_subfolder / "mobility.urdf"
    if not urdf_path.exists():
        print(f"[WARN] No mobility.urdf found in {input_subfolder}, skipping")
        return []

    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = str(input_subfolder.absolute())

    # Find all revolute joints
    revolute_joints = find_all_revolute_joints(root)

    if len(revolute_joints) == 0:
        print(f"[WARN] No revolute joints found in {subfolder_name}, skipping")
        return []

    print(f"[INFO] Found {len(revolute_joints)} revolute joints:")

    # Get max angle for each joint
    joint_max_angles = {}
    joint_mid_angles = {}
    for joint in revolute_joints:
        joint_name = joint.attrib.get('name', 'unnamed')
        parent_link = joint.find('parent').attrib['link']
        child_link = joint.find('child').attrib['link']
        max_angle = get_joint_max_angle(joint)
        mid_angle = max_angle / 2.0  # Mid point between 0 and max
        joint_max_angles[joint_name] = max_angle
        joint_mid_angles[joint_name] = mid_angle
        print(f"  - {joint_name}: {parent_link} -> {child_link}, max={math.degrees(max_angle):.2f}°, mid={math.degrees(mid_angle):.2f}°")

    # Compute normalization from angle=0 reference
    print(f"[INFO] Computing normalization from all-joints-at-0 reference pose...")
    joint_angles_ref = {joint.attrib.get('name', 'unnamed'): 0.0 for joint in revolute_joints}
    scene_ref = apply_multi_joint_fk(root, urdf_dir, joint_angles_ref, unit_scale)

    bbox_ref = scene_ref.bounding_box
    fixed_translation = -bbox_ref.centroid
    fixed_scale = 2.0 / bbox_ref.primitive.extents.max()

    print(f"[INFO] Normalization: translation={fixed_translation}, scale={fixed_scale:.4f}")

    # Find corresponding folder in output directory
    source_folder = output_base / subfolder_name
    if not source_folder.exists():
        print(f"[WARN] Output folder {source_folder} does not exist, skipping")
        return []

    # Load original config to get num_parts
    original_config = None
    if source_config_path and source_config_path.exists():
        with open(source_config_path, 'r') as f:
            configs = json.load(f)
            for config in configs:
                if config['file'] == subfolder_name:
                    original_config = config
                    break

    if original_config is None:
        print(f"[WARN] No config found for {subfolder_name}, using default num_parts=2")
        num_parts = 2  # Default value
    else:
        num_parts = original_config['num_parts']

    # Process both max and mid variants
    variants = [
        ("max", joint_max_angles),
        ("mid", joint_mid_angles)
    ]

    new_config_entries = []

    for variant_name, joint_angles in variants:
        print(f"\n[INFO] Rendering at {variant_name} joint angles...")
        scene = apply_multi_joint_fk(root, urdf_dir, joint_angles, unit_scale)

        # Create temporary output path for rendered image
        temp_output_path = input_subfolder / f"temp_rendering_{variant_name}.png"
        render_scene(scene, str(temp_output_path),
                    translation=fixed_translation,
                    scale=fixed_scale,
                    azimuth=azimuth,
                    elevation=elevation)

        # Create new folder with variant suffix
        target_folder = output_base / f"{subfolder_name}_{variant_name}"
        target_folder_name = f"{subfolder_name}_{variant_name}"
        print(f"[INFO] Copying {source_folder} to {target_folder}...")

        # Remove target if it exists
        if target_folder.exists():
            shutil.rmtree(target_folder)

        # Copy folder
        shutil.copytree(source_folder, target_folder)

        # Move rendered image to target folder as rendering.png
        target_image_path = target_folder / "rendering.png"
        shutil.move(str(temp_output_path), str(target_image_path))

        print(f"[SUCCESS] Created {target_folder} with new rendering.png")

        # Create new config entry for this variant
        # Use relative paths based on output directory structure
        output_dir_name = output_base.name
        new_entry = {
            "file": target_folder_name,
            "num_parts": num_parts,
            "valid": True,
            "mesh_path": f"mesh/partnet-mobility-preprocess/{subfolder_name}/voxel.glb",
            "surface_path": f"{output_dir_name}/{target_folder_name}/points.npy",
            "image_path": f"{output_dir_name}/{target_folder_name}/rendering.png",
            "iou_mean": 0.0,
            "iou_max": 0.0
        }
        new_config_entries.append(new_entry)

    return new_config_entries


def main():
    parser = argparse.ArgumentParser(
        description="Batch render URDF files at maximum and mid joint angles and create augmented dataset copies")
    parser.add_argument("--input", type=str, required=True,
                       help="Input folder containing subfolders with mobility.urdf files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output folder where processed subfolders will be copied with _max and _mid suffixes")
    parser.add_argument("--unit-scale", type=float, default=1.0,
                       help="Scale meshes by this factor")
    parser.add_argument("--azimuth", type=float, default=0.0,
                       help="Camera azimuth angle in degrees (0=+Z front, 90=+X right, 180=-Z back, 270=-X left)")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Camera elevation angle in degrees (0=horizontal, 90=top, -90=bottom)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    # Get all subfolders in input directory
    subfolders = [f for f in input_dir.iterdir() if f.is_dir()]

    if not subfolders:
        print(f"[WARN] No subfolders found in {input_dir}")
        return

    print(f"[INFO] Found {len(subfolders)} subfolders in {input_dir}")
    print(f"[INFO] Output base directory: {output_dir}")
    print(f"[INFO] Camera: azimuth={args.azimuth}°, elevation={args.elevation}°")

    # Look for source config in output directory
    source_config_path = output_dir / "object_part_configs.json"
    if not source_config_path.exists():
        # Fallback to default location
        source_config_path = Path("preprocessed_data/object_part_configs.json")

    # Process each subfolder
    success_count = 0
    all_new_entries = []

    for subfolder in sorted(subfolders):
        try:
            new_entries = process_single_object(
                subfolder, output_dir, source_config_path,
                args.unit_scale, args.azimuth, args.elevation
            )
            if new_entries:
                all_new_entries.extend(new_entries)
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {subfolder.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Update object_part_configs.json in OUTPUT directory with all new entries
    if all_new_entries:
        config_path = output_dir / "object_part_configs.json"
        update_object_part_configs(config_path, all_new_entries)

    print(f"\n{'='*80}")
    print(f"[DONE] Successfully processed {success_count}/{len(subfolders)} objects")
    print(f"[DONE] Added {len(all_new_entries)} new entries to {config_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy/convert URDF mesh references into a local mesh directory and rewrite paths."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import unquote, urlparse


def _resolve_mesh_path(filename: str, urdf_dir: Path) -> Path:
    parsed = urlparse(filename)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()

    normalized = filename
    if normalized.startswith("package://"):
        normalized = normalized.replace("package://", "", 1)

    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (urdf_dir / path).resolve()


def _unique_target_path(source: Path, mesh_dir: Path, used_names: set[str], suffix: str | None = None) -> Path:
    candidate_name = source.with_suffix(suffix).name if suffix else source.name
    if candidate_name not in used_names:
        used_names.add(candidate_name)
        return mesh_dir / candidate_name

    stem = source.stem
    target_suffix = suffix if suffix is not None else source.suffix
    index = 1
    while True:
        candidate_name = f"{stem}_{index}{target_suffix}"
        if candidate_name not in used_names:
            used_names.add(candidate_name)
            return mesh_dir / candidate_name
        index += 1


def _mesh_filename_for_urdf(target: Path, output_urdf: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return target.resolve().as_posix()
    return Path(os.path.relpath(target.resolve(), output_urdf.resolve().parent)).as_posix()


def _normalize_mesh_format(mesh_format: str | None) -> str | None:
    if mesh_format is None:
        return None
    normalized = mesh_format.lower().lstrip(".")
    if normalized in ("", "keep", "copy", "none"):
        return None
    if normalized not in ("obj", "stl"):
        raise ValueError("--mesh-format must be one of: keep, obj, stl")
    return normalized


def _mime_suffix(mime_type: str | None, fallback: str = ".png") -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/webp":
        return ".webp"
    return fallback


def _component_format(component_type: int) -> tuple[str, int, bool]:
    formats = {
        5120: ("b", 1, True),
        5121: ("B", 1, False),
        5122: ("h", 2, True),
        5123: ("H", 2, False),
        5125: ("I", 4, False),
        5126: ("f", 4, True),
    }
    if component_type not in formats:
        raise ValueError(f"Unsupported glTF component type: {component_type}")
    return formats[component_type]


def _accessor_dims(accessor_type: str) -> int:
    dims = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    if accessor_type not in dims:
        raise ValueError(f"Unsupported glTF accessor type: {accessor_type}")
    return dims[accessor_type]


def _decode_data_uri(uri: str) -> bytes:
    header, data = uri.split(",", 1)
    if ";base64" in header:
        return base64.b64decode(data)
    return unquote(data).encode("utf-8")


def _gltf_buffers(gltf: Any, gltf_path: Path) -> list[bytes]:
    blob = gltf.binary_blob() or b""
    buffers = []
    for buffer in gltf.buffers or []:
        uri = getattr(buffer, "uri", None)
        if not uri:
            buffers.append(blob)
        elif uri.startswith("data:"):
            buffers.append(_decode_data_uri(uri))
        else:
            buffers.append((gltf_path.parent / unquote(uri)).read_bytes())
    return buffers


def _read_accessor(gltf: Any, buffers: list[bytes], accessor_index: int) -> list[list[float | int]]:
    accessor = gltf.accessors[accessor_index]
    if getattr(accessor, "sparse", None) is not None:
        raise ValueError("Sparse glTF accessors are not supported by the OBJ exporter")
    if accessor.bufferView is None:
        raise ValueError("glTF accessor without bufferView is not supported")

    view = gltf.bufferViews[accessor.bufferView]
    raw = buffers[view.buffer]
    dims = _accessor_dims(accessor.type)
    fmt, component_size, _signed_or_float = _component_format(accessor.componentType)
    item_size = component_size * dims
    stride = view.byteStride or item_size
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    unpack = struct.Struct("<" + fmt * dims).unpack_from

    values = []
    for item_index in range(accessor.count):
        item_offset = start + item_index * stride
        item = list(unpack(raw, item_offset))
        if getattr(accessor, "normalized", False) and accessor.componentType != 5126:
            item = [_normalize_accessor_value(v, accessor.componentType) for v in item]
        values.append(item)
    return values


def _normalize_accessor_value(value: float | int, component_type: int) -> float:
    if component_type == 5120:
        return max(float(value) / 127.0, -1.0)
    if component_type == 5121:
        return float(value) / 255.0
    if component_type == 5122:
        return max(float(value) / 32767.0, -1.0)
    if component_type == 5123:
        return float(value) / 65535.0
    if component_type == 5125:
        return float(value) / 4294967295.0
    return float(value)


def _node_local_matrix(node: Any) -> list[list[float]]:
    if getattr(node, "matrix", None):
        values = list(node.matrix)
        return [[float(values[col * 4 + row]) for col in range(4)] for row in range(4)]

    translation = list(getattr(node, "translation", None) or [0.0, 0.0, 0.0])
    rotation = list(getattr(node, "rotation", None) or [0.0, 0.0, 0.0, 1.0])
    scale = list(getattr(node, "scale", None) or [1.0, 1.0, 1.0])

    x, y, z, w = [float(v) for v in rotation]
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]

    matrix = _identity_matrix()
    for row in range(3):
        for col in range(3):
            matrix[row][col] = rot[row][col] * float(scale[col])
    matrix[0][3] = float(translation[0])
    matrix[1][3] = float(translation[1])
    matrix[2][3] = float(translation[2])
    return matrix


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _transform_point(matrix: list[list[float]], point: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = [float(v) for v in point]
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _transform_normal(matrix: list[list[float]], normal: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = [float(v) for v in normal]
    nx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
    ny = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
    nz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def _texture_transform(texture_info: Any) -> Dict[str, Any]:
    extensions = getattr(texture_info, "extensions", None) or {}
    transform = extensions.get("KHR_texture_transform") if isinstance(extensions, dict) else None
    if not transform:
        return {}
    return {
        "offset": transform.get("offset", [0.0, 0.0]),
        "scale": transform.get("scale", [1.0, 1.0]),
        "rotation": transform.get("rotation", 0.0),
    }


def _apply_texture_transform(uv: list[float], transform: Dict[str, Any]) -> tuple[float, float]:
    u = float(uv[0])
    v = float(uv[1])
    scale = transform.get("scale", [1.0, 1.0])
    offset = transform.get("offset", [0.0, 0.0])
    rotation = float(transform.get("rotation", 0.0) or 0.0)

    u *= float(scale[0])
    v *= float(scale[1])
    if rotation:
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        u, v = cos_r * u - sin_r * v, sin_r * u + cos_r * v
    return u + float(offset[0]), v + float(offset[1])


def _safe_material_name(index: int | None) -> str:
    return f"material_{index if index is not None else 'default'}"


def _write_image(gltf: Any, buffers: list[bytes], gltf_path: Path, image_index: int, output_dir: Path, prefix: str) -> str:
    image = gltf.images[image_index]
    uri = getattr(image, "uri", None)

    if uri and uri.startswith("data:"):
        data = _decode_data_uri(uri)
        suffix = _mime_suffix(getattr(image, "mimeType", None))
    elif uri:
        source = gltf_path.parent / unquote(uri)
        data = source.read_bytes()
        suffix = source.suffix or _mime_suffix(getattr(image, "mimeType", None))
    else:
        if image.bufferView is None:
            raise ValueError(f"glTF image {image_index} has neither uri nor bufferView")
        view = gltf.bufferViews[image.bufferView]
        raw = buffers[view.buffer]
        start = view.byteOffset or 0
        end = start + view.byteLength
        data = raw[start:end]
        suffix = _mime_suffix(getattr(image, "mimeType", None))

    file_name = f"{prefix}_image_{image_index}{suffix}"
    (output_dir / file_name).write_bytes(data)
    return file_name


def _material_info(
    gltf: Any,
    buffers: list[bytes],
    gltf_path: Path,
    material_index: int | None,
    output_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    if material_index is None or not gltf.materials:
        return {"name": _safe_material_name(material_index), "color": [0.8, 0.8, 0.8, 1.0], "texture": None, "uv_transform": {}}

    material = gltf.materials[material_index]
    pbr = getattr(material, "pbrMetallicRoughness", None)
    color = list(getattr(pbr, "baseColorFactor", None) or [0.8, 0.8, 0.8, 1.0])
    texture_file = None
    uv_transform: Dict[str, Any] = {}
    texture_info = getattr(pbr, "baseColorTexture", None) if pbr is not None else None
    if texture_info is not None:
        texture = gltf.textures[texture_info.index]
        if texture.source is not None:
            texture_file = _write_image(gltf, buffers, gltf_path, texture.source, output_dir, prefix)
        uv_transform = _texture_transform(texture_info)
    return {
        "name": _safe_material_name(material_index),
        "color": color,
        "texture": texture_file,
        "uv_transform": uv_transform,
    }


def _write_obj_mtl(materials: Dict[int | None, Dict[str, Any]], mtl_path: Path) -> None:
    lines = []
    for info in materials.values():
        color = info["color"]
        lines.append(f"newmtl {info['name']}")
        lines.append(f"Kd {float(color[0]):.8f} {float(color[1]):.8f} {float(color[2]):.8f}")
        lines.append("Ka 0.00000000 0.00000000 0.00000000")
        lines.append("Ks 0.00000000 0.00000000 0.00000000")
        lines.append(f"d {float(color[3]) if len(color) > 3 else 1.0:.8f}")
        lines.append("illum 2")
        if info["texture"]:
            lines.append(f"map_Kd {info['texture']}")
        lines.append("")
    mtl_path.write_text("\n".join(lines), encoding="utf-8")


def _iter_scene_nodes(gltf: Any) -> Iterable[tuple[int, list[list[float]]]]:
    scene_index = gltf.scene if gltf.scene is not None else 0
    scene = gltf.scenes[scene_index]

    def walk(node_index: int, parent_matrix: list[list[float]]):
        node = gltf.nodes[node_index]
        world_matrix = _matmul(parent_matrix, _node_local_matrix(node))
        yield node_index, world_matrix
        for child_index in getattr(node, "children", None) or []:
            yield from walk(child_index, world_matrix)

    for root_node_index in scene.nodes or []:
        yield from walk(root_node_index, _identity_matrix())


def _primitive_indices(gltf: Any, buffers: list[bytes], primitive: Any, vertex_count: int) -> list[int]:
    if primitive.indices is None:
        return list(range(vertex_count))
    values = _read_accessor(gltf, buffers, primitive.indices)
    return [int(item[0]) for item in values]


def _write_obj_from_gltf(source: Path, target: Path, *, flip_uv_v: bool = True) -> None:
    try:
        from pygltflib import GLTF2
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("pygltflib is required for UV-preserving GLB to OBJ export") from exc

    gltf = GLTF2().load(str(source))
    buffers = _gltf_buffers(gltf, source)
    obj_lines = [f"mtllib {target.with_suffix('.mtl').name}", ""]
    materials: Dict[int | None, Dict[str, Any]] = {}
    vertex_offset = 0
    uv_offset = 0
    normal_offset = 0
    wrote_faces = 0

    for node_index, world_matrix in _iter_scene_nodes(gltf):
        node = gltf.nodes[node_index]
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        node_name = getattr(node, "name", None) or f"node_{node_index}"
        obj_lines.append(f"o {node_name}")

        for primitive_index, primitive in enumerate(mesh.primitives):
            if getattr(primitive, "mode", 4) not in (None, 4):
                continue

            attributes = primitive.attributes
            position_accessor = getattr(attributes, "POSITION", None)
            normal_accessor = getattr(attributes, "NORMAL", None)
            uv_accessor = getattr(attributes, "TEXCOORD_0", None)
            if position_accessor is None:
                continue

            positions = _read_accessor(gltf, buffers, position_accessor)
            normals = _read_accessor(gltf, buffers, normal_accessor) if normal_accessor is not None else []
            uvs = _read_accessor(gltf, buffers, uv_accessor) if uv_accessor is not None else []
            indices = _primitive_indices(gltf, buffers, primitive, len(positions))

            material_index = primitive.material
            if material_index not in materials:
                materials[material_index] = _material_info(gltf, buffers, source, material_index, target.parent, target.stem)
            material = materials[material_index]

            obj_lines.append(f"g {node_name}_primitive_{primitive_index}")
            obj_lines.append(f"usemtl {material['name']}")
            for position in positions:
                x, y, z = _transform_point(world_matrix, position[:3])
                obj_lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")

            uv_transform = material["uv_transform"]
            for uv in uvs:
                u, v = _apply_texture_transform(uv, uv_transform)
                if flip_uv_v:
                    v = 1.0 - v
                obj_lines.append(f"vt {u:.8f} {v:.8f}")

            for normal in normals:
                nx, ny, nz = _transform_normal(world_matrix, normal[:3])
                obj_lines.append(f"vn {nx:.8f} {ny:.8f} {nz:.8f}")

            for i in range(0, len(indices) - 2, 3):
                face = []
                for index in indices[i : i + 3]:
                    v_index = vertex_offset + index + 1
                    vt_index = uv_offset + index + 1 if uvs else None
                    vn_index = normal_offset + index + 1 if normals else None
                    if vt_index is not None and vn_index is not None:
                        face.append(f"{v_index}/{vt_index}/{vn_index}")
                    elif vt_index is not None:
                        face.append(f"{v_index}/{vt_index}")
                    elif vn_index is not None:
                        face.append(f"{v_index}//{vn_index}")
                    else:
                        face.append(str(v_index))
                obj_lines.append("f " + " ".join(face))
                wrote_faces += 1

            vertex_offset += len(positions)
            uv_offset += len(uvs)
            normal_offset += len(normals)
            obj_lines.append("")

    if wrote_faces == 0:
        raise ValueError(f"No triangle primitives were exported from {source}")

    target.write_text("\n".join(obj_lines), encoding="utf-8")
    _write_obj_mtl(materials, target.with_suffix(".mtl"))


def _write_mesh(source: Path, target: Path, mesh_format: str | None, flip_uv_v: bool = True) -> None:
    if mesh_format is None:
        shutil.copy2(source, target)
        return

    if mesh_format == "obj" and source.suffix.lower() in (".glb", ".gltf"):
        _write_obj_from_gltf(source, target, flip_uv_v=flip_uv_v)
        return

    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("trimesh is required when --mesh-format is used") from exc

    mesh = trimesh.load(str(source), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {source}")
    mesh.export(str(target), file_type=mesh_format)


def _add_collision_meshes_from_visuals(root: ET.Element) -> int:
    added = 0
    for link in root.findall("link"):
        if link.findall("collision"):
            continue
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or geometry.find("mesh") is None:
                continue

            collision = ET.Element("collision")
            origin = visual.find("origin")
            if origin is not None:
                collision.append(copy.deepcopy(origin))
            collision.append(copy.deepcopy(geometry))
            link.append(collision)
            added += 1
    return added


def localize_urdf_meshes(
    urdf_path: str | os.PathLike[str],
    output_urdf_path: str | os.PathLike[str] | None = None,
    mesh_dir: str | os.PathLike[str] = "meshes",
    *,
    absolute_paths: bool = False,
    mesh_format: str | None = None,
    add_collisions: bool = False,
    flip_uv_v: bool = True,
) -> Dict[str, Any]:
    """Copy or convert meshes referenced by a URDF and rewrite mesh filenames.

    Relative source mesh paths are resolved from the input URDF directory. If
    ``output_urdf_path`` is omitted, the input URDF is rewritten in place.
    Relative ``mesh_dir`` values are resolved from the output URDF directory.
    """
    mesh_format = _normalize_mesh_format(mesh_format)
    input_urdf = Path(urdf_path).resolve()
    output_urdf = Path(output_urdf_path).resolve() if output_urdf_path else input_urdf
    input_urdf_dir = input_urdf.parent

    mesh_dir_path = Path(mesh_dir)
    if not mesh_dir_path.is_absolute():
        mesh_dir_path = output_urdf.parent / mesh_dir_path
    mesh_dir_path = mesh_dir_path.resolve()
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    copied_by_source: Dict[tuple[Path, str | None], Path] = {}
    used_names: set[str] = set()
    missing: list[str] = []
    rewritten = 0
    written = 0
    target_suffix = f".{mesh_format}" if mesh_format is not None else None

    for mesh_el in root.findall(".//mesh"):
        filename = mesh_el.attrib.get("filename")
        if not filename:
            continue

        source = _resolve_mesh_path(filename, input_urdf_dir)
        if not source.exists():
            missing.append(f"{filename} -> {source}")
            continue

        source_key = (source, mesh_format)
        target = copied_by_source.get(source_key)
        if target is None:
            if mesh_format is None and source.parent.resolve() == mesh_dir_path:
                target = source
                used_names.add(target.name)
            else:
                target = _unique_target_path(source, mesh_dir_path, used_names, suffix=target_suffix)
                _write_mesh(source, target, mesh_format, flip_uv_v=flip_uv_v)
                written += 1
            copied_by_source[source_key] = target

        mesh_el.attrib["filename"] = _mesh_filename_for_urdf(target, output_urdf, absolute_paths)
        rewritten += 1

    added_collision_meshes = _add_collision_meshes_from_visuals(root) if add_collisions else 0

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Some URDF mesh references were not found:\n{joined}")

    ET.indent(tree, space="  ", level=0)
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

    return {
        "input_urdf": str(input_urdf),
        "output_urdf": str(output_urdf),
        "mesh_dir": str(mesh_dir_path),
        "localized_meshes": len(copied_by_source),
        "copied_meshes": written if mesh_format is None else 0,
        "converted_meshes": written if mesh_format is not None else 0,
        "rewritten_mesh_references": rewritten,
        "added_collision_meshes": added_collision_meshes,
        "mesh_format": mesh_format or "keep",
        "flip_uv_v": flip_uv_v,
        "mesh_files": [target.name for target in copied_by_source.values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy or convert URDF mesh references into a local meshes directory.")
    parser.add_argument("--urdf", required=True, help="Input URDF path")
    parser.add_argument("--output-urdf", default=None, help="Output URDF path. Default: rewrite --urdf in place")
    parser.add_argument("--mesh-dir", default="meshes", help="Directory for copied meshes. Relative to output URDF dir")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths into the output URDF")
    parser.add_argument(
        "--mesh-format",
        default="keep",
        choices=["keep", "obj", "stl"],
        help="Keep original mesh format or convert mesh files. OBJ/STL are safer for Isaac Sim URDF import",
    )
    parser.add_argument(
        "--no-flip-uv-v",
        action="store_true",
        help="Do not convert glTF top-left texture coordinates to OBJ-style V coordinates",
    )
    parser.add_argument("--add-collisions", action="store_true", help="Add collision meshes copied from visuals when absent")
    args = parser.parse_args()

    summary = localize_urdf_meshes(
        urdf_path=args.urdf,
        output_urdf_path=args.output_urdf,
        mesh_dir=args.mesh_dir,
        absolute_paths=args.absolute_paths,
        mesh_format=args.mesh_format,
        add_collisions=args.add_collisions,
        flip_uv_v=not args.no_flip_uv_v,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

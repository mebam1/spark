# clean_mesh.py
import argparse
import numpy as np
import trimesh
from typing import Optional


def load_as_mesh(path: str) -> trimesh.Trimesh:
    """
    Load mesh or scene. If scene, concatenate all valid geometries into one mesh.
    """
    obj = trimesh.load(path, force=None, process=False)

    if isinstance(obj, trimesh.Scene):
        meshes = []
        for g in obj.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0 and len(g.faces) > 0:
                meshes.append(g)
        if not meshes:
            raise ValueError("No valid mesh geometry found in the file.")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        # Some formats may load as PointCloud or Path3D, etc.
        raise ValueError(f"Unsupported loaded type: {type(obj)} (expected Trimesh/Scene).")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces. (Maybe it's a point cloud?)")

    return mesh


def remove_invalid_vertices(mesh: trimesh.Trimesh) -> None:
    """
    Remove NaN/Inf vertices and any faces referencing them.
    """
    valid_v = np.all(np.isfinite(mesh.vertices), axis=1)
    if np.all(valid_v):
        return
    valid_f = np.all(valid_v[mesh.faces], axis=1)
    mesh.update_faces(valid_f)
    mesh.remove_unreferenced_vertices()


def merge_vertices_by_tol(mesh: trimesh.Trimesh, tol: float) -> None:
    """
    Merge near-duplicate vertices in a version-safe way.
    Prefer merge_vertices(digits_vertex=...), fallback to rounding + merge.
    """
    if tol is None or tol <= 0:
        return

    digits = int(max(1, -np.log10(tol)))

    # Try modern signature
    try:
        mesh.merge_vertices(
            digits_vertex=digits,
            merge_norm=False,
            merge_tex=False,
        )
        return
    except TypeError:
        pass
    except Exception:
        pass

    # Fallback: round coords (may slightly quantize geometry, but works everywhere)
    mesh.vertices = np.round(mesh.vertices, decimals=digits)
    mesh.merge_vertices()


def remove_duplicate_faces_safe(mesh: trimesh.Trimesh) -> None:
    """
    Remove duplicate faces. Works across trimesh versions.
    Newer trimesh recommends update_faces(unique_faces()).
    """
    if hasattr(mesh, "unique_faces"):
        mask = mesh.unique_faces()
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        return

    # Fallbacks
    try:
        # Deprecated in some versions but may exist
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()


def remove_degenerate_faces_safe(mesh: trimesh.Trimesh) -> None:
    """
    Remove degenerate faces. Works across trimesh versions.
    Prefer update_faces(nondegenerate_faces()) when available.
    """
    if hasattr(mesh, "nondegenerate_faces"):
        mask = mesh.nondegenerate_faces()
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        return

    # Fallbacks
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()


def remove_zero_area_faces(mesh: trimesh.Trimesh, area_eps: float) -> None:
    """
    Remove faces with very small area (degenerate / zero-area).
    """
    if area_eps is None or area_eps <= 0:
        return

    try:
        areas = mesh.area_faces  # works for triangles; trimesh will triangulate where needed
        keep = areas > area_eps
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    except Exception:
        # Fallback: degenerate removal
        remove_degenerate_faces_safe(mesh)


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Keep only the largest connected component (by surface area) to drop floating debris.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: float(m.area))


def simplify_mesh_open3d(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    Quadric decimation using open3d. Also runs open3d cleanup passes to reduce junk triangles.
    """
    if target_faces is None or target_faces <= 0:
        return mesh
    if len(mesh.faces) <= target_faces:
        return mesh

    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )

    # Strong cleanup before simplify
    try:
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_non_manifold_edges()
    except Exception:
        pass

    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

    # Cleanup after simplify
    try:
        simplified.remove_degenerate_triangles()
        simplified.remove_duplicated_triangles()
        simplified.remove_duplicated_vertices()
        simplified.remove_non_manifold_edges()
    except Exception:
        pass

    out = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )
    return out


def clean_mesh(
    in_path: str,
    out_path: str,
    merge_tol: float = 1e-6,
    area_eps: float = 1e-12,
    keep_largest: bool = False,
    target_faces: int = 0,
):
    mesh = load_as_mesh(in_path)

    print("==== BEFORE ====")
    print(f"Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)} | Watertight: {mesh.is_watertight}")

    # 0) remove NaN/Inf
    remove_invalid_vertices(mesh)

    # 1) merge near-duplicate vertices
    merge_vertices_by_tol(mesh, merge_tol)

    # 2) remove duplicate + degenerate faces
    remove_duplicate_faces_safe(mesh)
    remove_degenerate_faces_safe(mesh)

    # 3) remove zero-area faces (extra pass)
    remove_zero_area_faces(mesh, area_eps)

    # 4) final vertex cleanup
    mesh.remove_unreferenced_vertices()

    # 5) optionally keep only largest component
    if keep_largest:
        mesh = keep_largest_component(mesh)

    # 6) simplify (ONLY open3d; do not call trimesh.simplify_quadratic_decimation)
    if target_faces and target_faces > 0:
        try:
            before_f = len(mesh.faces)
            mesh = simplify_mesh_open3d(mesh, target_faces)
            mesh.remove_unreferenced_vertices()
            print(f"[OK] Simplified from {before_f} to {len(mesh.faces)} faces")
        except ImportError:
            print("[WARN] open3d not installed; skipping simplification. Install: pip install open3d")
        except Exception as e:
            print(f"[WARN] open3d simplification failed: {e}")

    # 7) final cleanup pass
    merge_vertices_by_tol(mesh, merge_tol)
    remove_duplicate_faces_safe(mesh)
    remove_degenerate_faces_safe(mesh)
    remove_zero_area_faces(mesh, area_eps)
    mesh.remove_unreferenced_vertices()

    print("==== AFTER ====")
    print(f"Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)} | Watertight: {mesh.is_watertight}")

    mesh.export(out_path)
    print(f"[OK] Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_mesh", type=str, required=True, help="input mesh path (.obj/.ply/.stl/.glb/...)")
    parser.add_argument("--out_mesh", type=str, required=True, help="output mesh path")
    parser.add_argument("--merge_tol", type=float, default=1e-6, help="vertex merge tolerance (~1e-6 to 1e-4)")
    parser.add_argument("--area_eps", type=float, default=1e-12, help="remove faces with area <= eps")
    parser.add_argument("--keep_largest", action="store_true", help="keep only largest connected component")
    parser.add_argument("--target_faces", type=int, default=0, help="simplify to target face count (0 disables)")
    args = parser.parse_args()

    clean_mesh(
        in_path=args.in_mesh,
        out_path=args.out_mesh,
        merge_tol=args.merge_tol,
        area_eps=args.area_eps,
        keep_largest=args.keep_largest,
        target_faces=args.target_faces,
    )


if __name__ == "__main__":
    main()

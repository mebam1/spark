import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from URDFoptimizer.render.localize_urdf_meshes import localize_urdf_meshes


class LocalizeUrdfMeshesTests(unittest.TestCase):
    def test_localize_urdf_meshes_copies_meshes_and_rewrites_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_mesh_dir = root / "split_parts"
            urdf_dir = root / "articulation"
            source_mesh_dir.mkdir()
            urdf_dir.mkdir()

            (source_mesh_dir / "body.glb").write_bytes(b"body")
            (source_mesh_dir / "door.glb").write_bytes(b"door")
            urdf_path = urdf_dir / "mobility_refined.urdf"
            urdf_path.write_text(
                """<?xml version="1.0"?>
<robot name="object">
  <link name="base"/>
  <link name="link0">
    <visual>
      <geometry><mesh filename="../split_parts/body.glb"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="../split_parts/body.glb"/></geometry>
    </collision>
  </link>
  <link name="link1">
    <visual>
      <geometry><mesh filename="../split_parts/door.glb"/></geometry>
    </visual>
  </link>
</robot>
""",
                encoding="utf-8",
            )

            summary = localize_urdf_meshes(urdf_path, mesh_dir="meshes")

            self.assertEqual(summary["localized_meshes"], 2)
            self.assertEqual(summary["copied_meshes"], 2)
            self.assertEqual(summary["rewritten_mesh_references"], 3)
            self.assertTrue((urdf_dir / "meshes" / "body.glb").exists())
            self.assertTrue((urdf_dir / "meshes" / "door.glb").exists())

            xml_root = ET.parse(urdf_path).getroot()
            filenames = [mesh.attrib["filename"] for mesh in xml_root.findall(".//mesh")]
            self.assertEqual(filenames, ["meshes/body.glb", "meshes/body.glb", "meshes/door.glb"])

            second_summary = localize_urdf_meshes(urdf_path, mesh_dir="meshes")
            self.assertEqual(second_summary["localized_meshes"], 2)
            self.assertEqual(second_summary["copied_meshes"], 0)
            self.assertEqual(second_summary["converted_meshes"], 0)
            self.assertEqual(sorted(path.name for path in (urdf_dir / "meshes").glob("*.glb")), ["body.glb", "door.glb"])

    def test_convert_mesh_format_rewrites_to_converted_files_and_adds_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_mesh_dir = root / "split_parts"
            urdf_dir = root / "articulation"
            source_mesh_dir.mkdir()
            urdf_dir.mkdir()

            (source_mesh_dir / "body.glb").write_bytes(b"body")
            urdf_path = urdf_dir / "mobility_refined.urdf"
            output_urdf = urdf_dir / "mobility_refined_isaac.urdf"
            urdf_path.write_text(
                """<?xml version="1.0"?>
<robot name="object">
  <link name="link0">
    <visual>
      <origin xyz="1 2 3" rpy="0 0 0"/>
      <geometry><mesh filename="../split_parts/body.glb"/></geometry>
    </visual>
  </link>
</robot>
""",
                encoding="utf-8",
            )

            def fake_write_mesh(source: Path, target: Path, mesh_format: str | None, flip_uv_v: bool = True) -> None:
                target.write_bytes(source.read_bytes())

            with patch("URDFoptimizer.render.localize_urdf_meshes._write_mesh", side_effect=fake_write_mesh):
                summary = localize_urdf_meshes(
                    urdf_path,
                    output_urdf_path=output_urdf,
                    mesh_dir="meshes",
                    mesh_format="obj",
                    add_collisions=True,
                )

            self.assertEqual(summary["mesh_format"], "obj")
            self.assertEqual(summary["flip_uv_v"], True)
            self.assertEqual(summary["converted_meshes"], 1)
            self.assertEqual(summary["copied_meshes"], 0)
            self.assertEqual(summary["added_collision_meshes"], 1)
            self.assertTrue((urdf_dir / "meshes" / "body.obj").exists())

            xml_root = ET.parse(output_urdf).getroot()
            filenames = [mesh.attrib["filename"] for mesh in xml_root.findall(".//mesh")]
            self.assertEqual(filenames, ["meshes/body.obj", "meshes/body.obj"])
            collision_origin = xml_root.find("./link[@name='link0']/collision/origin")
            self.assertIsNotNone(collision_origin)
            self.assertEqual(collision_origin.attrib["xyz"], "1 2 3")

            with patch("URDFoptimizer.render.localize_urdf_meshes._write_mesh", side_effect=fake_write_mesh):
                localize_urdf_meshes(
                    urdf_path,
                    output_urdf_path=output_urdf,
                    mesh_dir="meshes",
                    mesh_format="obj",
                    add_collisions=True,
                )
            self.assertEqual(sorted(path.name for path in (urdf_dir / "meshes").glob("*.obj")), ["body.obj"])

    def test_convert_mesh_format_can_disable_uv_v_flip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_mesh_dir = root / "split_parts"
            urdf_dir = root / "articulation"
            source_mesh_dir.mkdir()
            urdf_dir.mkdir()

            (source_mesh_dir / "body.glb").write_bytes(b"body")
            urdf_path = urdf_dir / "mobility.urdf"
            urdf_path.write_text(
                """<?xml version="1.0"?>
<robot name="object">
  <link name="link0">
    <visual>
      <geometry><mesh filename="../split_parts/body.glb"/></geometry>
    </visual>
  </link>
</robot>
""",
                encoding="utf-8",
            )

            seen_flip_values = []

            def fake_write_mesh(source: Path, target: Path, mesh_format: str | None, flip_uv_v: bool = True) -> None:
                seen_flip_values.append(flip_uv_v)
                target.write_bytes(source.read_bytes())

            with patch("URDFoptimizer.render.localize_urdf_meshes._write_mesh", side_effect=fake_write_mesh):
                summary = localize_urdf_meshes(
                    urdf_path,
                    output_urdf_path=urdf_dir / "mobility_isaac.urdf",
                    mesh_dir="meshes",
                    mesh_format="obj",
                    flip_uv_v=False,
                )

            self.assertEqual(summary["flip_uv_v"], False)
            self.assertEqual(seen_flip_values, [False])

    def test_localize_urdf_meshes_raises_for_missing_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            urdf_path = Path(tmp) / "mobility.urdf"
            urdf_path.write_text(
                """<?xml version="1.0"?>
<robot name="object">
  <link name="link0">
    <visual>
      <geometry><mesh filename="missing.glb"/></geometry>
    </visual>
  </link>
</robot>
""",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                localize_urdf_meshes(urdf_path)


if __name__ == "__main__":
    unittest.main()

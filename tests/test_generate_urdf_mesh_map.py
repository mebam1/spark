import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from VLMguidance.generate_urdf import generate_urdf_content


class GenerateUrdfMeshMapTests(unittest.TestCase):
    def test_mesh_map_list_writes_multiple_visuals_for_same_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_dir = root / "articulation"
            parts_dir = root / "split_parts"
            object_dir.mkdir()
            parts_dir.mkdir()

            metadata = {
                "object_name": "cabinet",
                "parts": [
                    {"label": "link0", "name": "body", "parent": "base", "joint_type": "fixed"},
                    {
                        "label": "link1",
                        "name": "door",
                        "parent": "link0",
                        "joint_type": "revolute",
                        "axis": "0 0 1",
                    },
                ],
            }
            mesh_map = {
                "link0": ["../split_parts/body_a.glb", "../split_parts/body_b.glb"],
                "link1": "../split_parts/door.glb",
            }

            urdf = generate_urdf_content(metadata, object_dir=object_dir, mesh_map=mesh_map)
            xml_root = ET.fromstring(urdf)

            link0 = xml_root.find("./link[@name='link0']")
            link1 = xml_root.find("./link[@name='link1']")

            self.assertIsNotNone(link0)
            self.assertIsNotNone(link1)
            self.assertEqual(
                [mesh.attrib["filename"] for mesh in link0.findall("./visual/geometry/mesh")],
                ["../split_parts/body_a.glb", "../split_parts/body_b.glb"],
            )
            self.assertEqual(
                [mesh.attrib["filename"] for mesh in link1.findall("./visual/geometry/mesh")],
                ["../split_parts/door.glb"],
            )

    def test_mesh_dir_paths_are_written_relative_to_urdf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_dir = root / "runs" / "articulation"
            mesh_dir = root / "runs" / "split_parts"
            object_dir.mkdir(parents=True)
            mesh_dir.mkdir(parents=True)

            metadata = {
                "object_name": "single",
                "parts": [{"label": "link0", "name": "body", "parent": "base", "joint_type": "fixed"}],
            }

            urdf = generate_urdf_content(
                metadata,
                object_dir=object_dir,
                mesh_dir=mesh_dir,
                mesh_pattern="part_{index:02d}.glb",
            )
            xml_root = ET.fromstring(urdf)
            mesh = xml_root.find("./link[@name='link0']/visual/geometry/mesh")

            self.assertIsNotNone(mesh)
            self.assertEqual(mesh.attrib["filename"], "../split_parts/part_00.glb")


if __name__ == "__main__":
    unittest.main()

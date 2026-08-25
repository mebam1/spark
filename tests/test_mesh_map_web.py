import json
import tempfile
import unittest
from pathlib import Path

from URDFoptimizer.render.mesh_map_web import MeshMapWebState


class MeshMapWebTests(unittest.TestCase):
    def test_save_assignments_groups_multiple_meshes_under_one_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata.json"
            split_dir = root / "parts"
            output_path = root / "mesh_map.json"
            split_dir.mkdir()

            metadata_path.write_text(
                json.dumps(
                    {
                        "object_name": "cabinet",
                        "parts": [
                            {"label": "link0", "name": "body", "parent": "base", "joint_type": "fixed"},
                            {"label": "link1", "name": "door", "parent": "link0", "joint_type": "revolute"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (split_dir / "part_0_body_a.glb").write_bytes(b"glb")
            (split_dir / "part_1_body_b.glb").write_bytes(b"glb")
            (split_dir / "part_2_door.glb").write_bytes(b"glb")
            (split_dir / "split_manifest.json").write_text(
                json.dumps(
                    [
                        {"index": 0, "file": "part_0_body_a.glb", "geometry_name": "body_a"},
                        {"index": 1, "file": "part_1_body_b.glb", "geometry_name": "body_b"},
                        {"index": 2, "file": "part_2_door.glb", "geometry_name": "door"},
                    ]
                ),
                encoding="utf-8",
            )

            state = MeshMapWebState(metadata_path, split_dir, output_path)
            summary = state.save_assignments(
                {
                    "part_0_body_a.glb": "link0",
                    "part_1_body_b.glb": "link0",
                    "part_2_door.glb": "link1",
                }
            )

            self.assertEqual(summary["links_without_mesh"], [])
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {
                    "link0": ["parts/part_0_body_a.glb", "parts/part_1_body_b.glb"],
                    "link1": "parts/part_2_door.glb",
                },
            )


if __name__ == "__main__":
    unittest.main()

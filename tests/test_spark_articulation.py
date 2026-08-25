import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from URDFoptimizer.spark_articulation.fk import compute_link_transforms
from URDFoptimizer.spark_articulation.model import DifferentiableArticulationModel, MeshPartTensors
from URDFoptimizer.spark_articulation.renderer import RendererConfig, build_silhouette_renderer, render_silhouette
from URDFoptimizer.spark_articulation.urdf_model import VisualSpec, parse_urdf


def _write_urdf(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8")
    with tmp:
        tmp.write(textwrap.dedent(text).strip())
    return Path(tmp.name)


def _single_joint_urdf() -> Path:
    return _write_urdf(
        """
        <?xml version="1.0"?>
        <robot name="single_joint">
          <link name="base"/>
          <link name="body"/>
          <link name="door">
            <visual>
              <geometry><mesh filename="door.glb"/></geometry>
              <origin xyz="0 0 0" rpy="0 0 0"/>
            </visual>
          </link>
          <joint name="base_to_body" type="fixed">
            <parent link="base"/>
            <child link="body"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
          </joint>
          <joint name="hinge" type="revolute">
            <parent link="body"/>
            <child link="door"/>
            <origin xyz="1 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="0" upper="1.5707963267948966" effort="1" velocity="1"/>
          </joint>
        </robot>
        """
    )


class SparkArticulationTests(unittest.TestCase):
    def test_single_revolute_joint_fk(self):
        urdf = parse_urdf(_single_joint_urdf())
        transforms = compute_link_transforms(
            urdf,
            joint_values={"hinge": torch.tensor(math.pi / 2)},
            device=torch.device("cpu"),
        )
        point = torch.tensor([[1.0, 0.0, 0.0]])
        point_h = torch.cat([point, torch.ones(1, 1)], dim=1)
        transformed = (transforms["door"] @ point_h.T).T[:, :3]
        self.assertTrue(torch.allclose(transformed[0], torch.tensor([1.0, 1.0, 0.0]), atol=1e-5))

    def test_origin_offset_preserves_zero_pose_and_changes_open_pivot(self):
        urdf = parse_urdf(_write_urdf(
            """
            <?xml version="1.0"?>
            <robot name="offset_joint">
              <link name="base"/>
              <link name="door"/>
              <joint name="hinge" type="revolute">
                <parent link="base"/>
                <child link="door"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="0" upper="1.5707963267948966" effort="1" velocity="1"/>
              </joint>
            </robot>
            """
        ))
        offset = {"hinge": torch.tensor([1.0, 0.0, 0.0])}
        closed = compute_link_transforms(urdf, joint_values={"hinge": 0.0}, origin_offsets=offset)
        opened = compute_link_transforms(urdf, joint_values={"hinge": math.pi / 2}, origin_offsets=offset)

        point = torch.tensor([[2.0, 0.0, 0.0]])
        point_h = torch.cat([point, torch.ones(1, 1)], dim=1)
        closed_point = (closed["door"] @ point_h.T).T[:, :3]
        opened_point = (opened["door"] @ point_h.T).T[:, :3]

        self.assertTrue(torch.allclose(closed_point[0], torch.tensor([2.0, 0.0, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(opened_point[0], torch.tensor([1.0, 1.0, 0.0]), atol=1e-5))

    def test_full_urdf_tree_fk(self):
        urdf = parse_urdf(_write_urdf(
            """
            <?xml version="1.0"?>
            <robot name="tree">
              <link name="base"/>
              <link name="a"/>
              <link name="b"/>
              <joint name="base_to_a" type="revolute">
                <parent link="base"/>
                <child link="a"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="0" upper="1.5707963267948966" effort="1" velocity="1"/>
              </joint>
              <joint name="a_to_b" type="revolute">
                <parent link="a"/>
                <child link="b"/>
                <origin xyz="1 0 0" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="0" upper="1.5707963267948966" effort="1" velocity="1"/>
              </joint>
            </robot>
            """
        ))
        transforms = compute_link_transforms(
            urdf,
            joint_values={"base_to_a": math.pi / 2, "a_to_b": math.pi / 2},
        )
        origin_h = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        b_origin = (transforms["b"] @ origin_h.T).T[:, :3]
        self.assertTrue(torch.allclose(b_origin[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5))

    def test_differentiable_rendering_and_joint_parameter_path(self):
        try:
            import pytorch3d  # noqa: F401
        except Exception:
            self.skipTest("PyTorch3D is not installed in this environment")

        urdf = parse_urdf(_single_joint_urdf())
        verts = torch.tensor(
            [
                [1.0, -0.2, 0.0],
                [2.0, -0.2, 0.0],
                [2.0, 0.2, 0.0],
                [1.0, 0.2, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        part = MeshPartTensors("door", VisualSpec(mesh_filename="door.glb"), verts, faces)
        model = DifferentiableArticulationModel(
            urdf,
            [part],
            optimize_joint_names=["hinge"],
            initial_joint_values={"hinge": math.pi / 4},
            auto_normalize=True,
        )
        renderer = build_silhouette_renderer(RendererConfig(image_size=32, faces_per_pixel=10), torch.device("cpu"))
        silhouette = render_silhouette(renderer, model())
        loss = silhouette.mean()
        loss.backward()
        self.assertIsNotNone(model.angle_deltas[0].grad)


if __name__ == "__main__":
    unittest.main()

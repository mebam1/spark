# SPARK Articulation-Only Baseline

This package implements the SPARK-faithful articulation reasoning and refinement
path for an existing textured, part-segmented mesh. It does not generate mesh
geometry or textures.

## Included

1. VLM coarse structure metadata via `VLMguidance/generate_json.py`
2. VLM joint type and axis re-prediction via `VLMguidance/repredict_axis.py`
3. Coarse URDF generation with existing mesh references via `VLMguidance/generate_urdf.py`
4. VLM open-state image generation via `VLMguidance/generate_prompt_open.py` and `generate_image_open.py`
5. Differentiable hierarchical URDF FK
6. PyTorch3D soft silhouette rendering with fixed camera parameters
7. Silhouette region overlap and edge alignment loss
8. Origin-offset and open-state angle regularization

## Excluded

- Diffusion Transformer mesh generation
- VAE latent generation
- DINOv2 conditioning
- Part-image-guided mesh synthesis
- Texture generation
- RGB photometric loss
- Multi-state supervision
- Joint type or axis gradient optimization
- Camera pose, camera intrinsics, or mesh vertex optimization

## Entrypoints

Optimize an existing URDF:

```bash
python URDFoptimizer/optimize_spark_articulation.py \
  --urdf path/to/mobility.urdf \
  --open-image path/to/open.png \
  --out-urdf path/to/mobility_refined.urdf
```

Run the articulation-only pipeline from an RGB image and existing segmented
meshes:

```bash
python run_articulation.py \
  --image image/example.png \
  --output-dir output/example_articulation \
  --mesh-dir path/to/segmented_meshes
```

Use `--mesh-map mesh_map.json` when semantic link labels should map to explicit
mesh files, for example:

```json
{
  "link0": "body.glb",
  "link1": "door.glb"
}
```

## Ambiguities

[Paper ambiguity] SPARK optimizes continuous joint parameters against an
open-state image, but a URDF pivot change can move the closed mesh if written
naively.

[Implementation choice] The differentiable FK and URDF writer preserve the
existing zero pose by compensating child-local visual, collision, and outgoing
joint origins when a joint origin offset is written.

[Reason] The user-provided mesh is already assembled and segmented. Preserving
the zero pose changes the rotation pivot without modifying vertices or breaking
the original closed-state alignment.

[Paper ambiguity] The optimized open-state angle is a pose variable, not a
standard URDF structural field.

[Implementation choice] The refined URDF stores optimized joint origins. The
learned open-state joint values are written to `optimization_summary.json`.

[Reason] This keeps URDF structure valid while recording the optimized pose used
for matching `I_open`.


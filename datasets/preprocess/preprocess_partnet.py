import os
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
def _limit_threads():
    # Prevent each process from spawning too many threads (avoid over-subscription)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    # If render.py runs in a headless environment, enable EGL; remove this line if a display is available
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

def _process_one(object_id, input_path, output_path):
    """
    Process one object: select voxel.glb/part.glb -> run 4 sub-scripts in order.
    Returns (object_id, True/False).
    """
    subfolder_path = os.path.join(input_path, object_id)

    voxel_glb_path = os.path.join(subfolder_path, 'voxel.glb')
    part_glb_path  = os.path.join(subfolder_path, 'part.glb')

    if os.path.exists(voxel_glb_path):
        parts_glb_path = voxel_glb_path
        mesh_type = "voxel"
    else:
        parts_glb_path = part_glb_path
        mesh_type = "part"

    print(f"[{object_id}] using {mesh_type}.glb for point sampling")

    # Four steps: mesh_to_point / render / render_notexture / render_part
    cmds = [
        ["python", "datasets/preprocess/mesh_to_point.py",
         "--input", parts_glb_path, "--output", output_path, "--name", object_id],
        ["python", "datasets/preprocess/render.py",
         "--input", subfolder_path, "--output", output_path, "--name", object_id],
        ["python", "datasets/preprocess/render_notexture.py",
         "--input", subfolder_path, "--output", output_path, "--name", object_id],
        ["python", "datasets/preprocess/render_part.py",
         "--input", subfolder_path, "--output", output_path, "--name", object_id],
    ]

    try:
        for cmd in cmds:
            # Use subprocess.run instead of os.system; raises exception on failure for easier capture
            subprocess.run(cmd, check=True)
        return (object_id, True)
    except subprocess.CalledProcessError as e:
        print(f"[{object_id}] step failed: {e}")
        return (object_id, False)
    except Exception as e:
        print(f"[{object_id}] unexpected error: {e}")
        return (object_id, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='mesh/partnet_glb', help='Input directory containing subfolders with part.glb and texture.glb')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel workers (default 1 = serial)')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get all subfolders (all have same format: part.glb + texture.glb)
    subfolders = [item for item in os.listdir(input_path) 
                  if os.path.isdir(os.path.join(input_path, item))]

    print(f"Found {len(subfolders)} subfolders")
    jobs = max(1, int(args.jobs))

    if jobs == 1:
        # Serial fallback (matches original behavior)
        for object_id in tqdm(subfolders):
            _process_one(object_id, input_path, output_path)
    else:
        print(f"Running with {jobs} parallel workers ...")
        ok_cnt = 0
        with ProcessPoolExecutor(max_workers=jobs, initializer=_limit_threads) as ex:
            futures = {
                ex.submit(_process_one, object_id, input_path, output_path): object_id
                for object_id in subfolders
            }
            for fut in tqdm(as_completed(futures), total=len(subfolders)):
                object_id = futures[fut]
                try:
                    oid, ok = fut.result()
                    ok_cnt += int(bool(ok))
                    if not ok:
                        print(f"[{oid}] failed")
                except Exception as e:
                    print(f"[{object_id}] error in worker: {e}")
        print(f"Done: {ok_cnt}/{len(subfolders)} succeeded")

    
    # generate configs
    configs = []
    for object_id in tqdm(subfolders):
        # Fixed structure: preprocessed_data/object_id/
        mesh_path = os.path.join(output_path, object_id)
        num_parts_path = os.path.join(mesh_path, 'num_parts.json')
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering.png')
        
        # Use voxel.glb if available, otherwise part.glb for training
        voxel_glb_path = os.path.join(input_path, object_id, 'voxel.glb')
        part_glb_path = os.path.join(input_path, object_id, 'part.glb')
        
        if os.path.exists(voxel_glb_path):
            parts_glb_path = voxel_glb_path
        else:
            parts_glb_path = part_glb_path
        
        config = {
            "file": object_id,
            "num_parts": 0,
            "valid": False,
            "mesh_path": parts_glb_path,  # Use part.glb for mesh_path
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,  # No IoU calculation
            "iou_max": 0.0    # No IoU calculation
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            continue
    
    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)
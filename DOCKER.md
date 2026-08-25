# SPARK Docker Guide

이 문서는 `README.md`의 로컬/conda 명령을 Docker Compose 기준으로 바꾼 실행 가이드입니다.

## 기본 전제

- Docker와 Docker Compose v2가 필요합니다.
- GPU 실행에는 NVIDIA driver와 NVIDIA Container Toolkit이 필요합니다.
- 저장소 루트 전체가 컨테이너의 `/workspace/SPARK`에 bind mount됩니다.
- `.env`, `image/`, `mesh/`, `pretrained_weights/`, `output/`, `results/` 등 루트 아래 파일 변경은 호스트와 컨테이너가 공유합니다.

## 빌드

```bash
docker compose build
```

기본 이미지는 CUDA 12.4 devel 환경이며 Python 3.10, PyTorch 2.5.1 CUDA 12.4 wheel, `torch-cluster`, `requirements.txt`, PyTorch3D를 설치합니다.

PyTorch3D CUDA extension을 특정 GPU 아키텍처로 빌드해야 하면 `TORCH_CUDA_ARCH_LIST`를 지정합니다.

```bash
TORCH_CUDA_ARCH_LIST="8.6;8.9" docker compose build
```

## 컨테이너 셸

```bash
docker compose run --rm spark bash
```

이미 실행 중인 컨테이너에 들어가려면 다음을 사용합니다.

```bash
docker compose exec spark bash
```

## CUDA 확인

```bash
docker compose run --rm spark nvcc --version
docker compose run --rm spark python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## API 키 설정

루트에 `.env` 파일을 만들고 필요한 키를 넣습니다.

```env
# OpenAI: metadata.json prediction, joint-axis correction, prompt generation
OPENAI_API_KEY=

# Google Gemini: part-image and open-state image generation
GEMINI_API_KEY=

# Meshy: texturing
MESHY_API_KEY=
```

Compose는 루트의 `.env` 값을 읽어 컨테이너 환경 변수로 전달합니다.

## Pretrained Weights

체크포인트는 첫 실행 시 `pretrained_weights/<name>/` 아래로 자동 다운로드됩니다. 미리 다운로드하려면:

```bash
docker compose run --rm spark python -m src.utils.download_weights
```

## Quick Start

입력 이미지를 호스트의 `./image/`에 넣고 전체 파이프라인을 실행합니다.

```bash
docker compose run --rm spark python run.py
```

`run.py` 상단의 `step1`부터 `step9` 플래그를 수정하면 개별 단계를 켜고 끌 수 있습니다. 루트가 mount되어 있으므로 수정 후 이미지를 다시 빌드할 필요는 없습니다.

## Demo UI

```bash
docker compose run --rm --service-ports spark python demo/app.py
```

브라우저에서 `http://localhost:7860`으로 접속합니다.

## Data Preprocessing

PartNet-Mobility 예시:

```bash
# Convert PartNet-Mobility (obj+mtl) to GLB
docker compose run --rm spark python datasets/merge_to_glb.py mesh/partnet_test mesh/partnet_glb

# Voxelize meshes
docker compose run --rm spark python datasets/voxel_surface.py mesh/partnet_glb --subfolder -r 200

# Build PartCrafter training data
docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 spark python datasets/preprocess/preprocess_partnet.py \
    --input mesh/partnet_glb --output preprocessed_data
```

## Training

```bash
docker compose run --rm spark bash scripts/train/train_partcrafter.sh --config configs/mp16_nt1024.yaml
```

특정 GPU만 사용하려면:

```bash
docker compose run --rm -e CUDA_VISIBLE_DEVICES=0,1 spark bash scripts/train/train_partcrafter.sh --config configs/mp16_nt1024.yaml
```

W&B를 사용하려면 `.env`에 `WANDB_API_KEY`를 추가합니다.

## Common Script Entrypoints

Single-image object inference:

```bash
docker compose run --rm spark python scripts/inference/object.py \
    --image_path image/example.png \
    --num_parts 4 \
    --output_dir results \
    --render
```

Per-part image inference:

```bash
docker compose run --rm spark python scripts/inference/part_images.py \
    --image_folder output/example_case \
    --output_dir results \
    --render
```

Render a generated mesh:

```bash
docker compose run --rm spark python scripts/render/glb.py \
    --input results/example/object.glb \
    --output_dir render_output \
    --gif
```

## HuggingFace Upload

로그인:

```bash
docker compose run --rm spark hf auth login
```

Xet backend 설치가 필요하면 실행 중 컨테이너 안에서 설치하거나, 일회성으로 다음처럼 실행합니다.

```bash
docker compose run --rm spark pip install hf_xet
```

업로드:

```bash
docker compose run --rm spark python scripts/tools/upload_weights.py
```

중단되면 같은 명령을 다시 실행하면 됩니다. 이미 업로드된 파일은 SHA 비교로 건너뜁니다.

"""
Upload pretrained_weights/ to the SPARK HuggingFace model repo.

Usage:
    huggingface-cli login        # one-time, paste a write token
    python scripts/tools/upload_weights.py
"""
from huggingface_hub import HfApi

REPO_ID = "2bidoubi/SPARK"
FOLDER = "pretrained_weights"

if __name__ == "__main__":
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
    api.upload_large_folder(
        folder_path=FOLDER,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=[
            "**/.cache/**",
            "**/__pycache__/**",
            "*.tmp",
        ],
    )
    print(f"Uploaded {FOLDER}/ -> https://huggingface.co/{REPO_ID}")

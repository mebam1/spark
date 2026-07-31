"""
Auto-download SPARK pretrained weights from HuggingFace on first use.

The full bundle (PartCrafter, TripoSG, RMBG-1.4) lives in
https://huggingface.co/2bidoubi/SPARK and is mirrored locally under
`pretrained_weights/<subfolder>/`.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "2bidoubi/SPARK"
WEIGHTS_ROOT = Path("pretrained_weights")


def _has_files(path: Path) -> bool:
    return path.exists() and any(p for p in path.rglob("*") if p.is_file())


def ensure_weights(subfolder: str) -> str:
    """Ensure `pretrained_weights/<subfolder>` exists; download if missing.

    Returns the local path as a string for direct use in `from_pretrained(...)`.
    """
    target = WEIGHTS_ROOT / subfolder
    if _has_files(target):
        return str(target)

    print(f"[SPARK] downloading {subfolder} from {HF_REPO_ID} -> {target}")
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(WEIGHTS_ROOT),
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"],
    )
    return str(target)


def ensure_all() -> None:
    """Download every weight folder. Useful as a one-shot prefetch."""
    for sub in ("PartCrafter", "TripoSG", "RMBG-1.4"):
        ensure_weights(sub)


if __name__ == "__main__":
    ensure_all()

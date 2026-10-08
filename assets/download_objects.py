"""下载并解压 RoboTwin2.0 的对象资源到仓库 assets/objects。"""

from __future__ import annotations

from pathlib import Path
import zipfile


REPOSITORY_ID = "TianxingChen/RoboTwin2.0"
ASSETS_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = ASSETS_DIR / "objects.zip"
OBJECTS_DIR = ASSETS_DIR / "objects"
COMPLETION_MARKER = OBJECTS_DIR / ".download_complete"


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise RuntimeError(f"拒绝解压仓库目录之外的路径: {member.filename}")
        archive.extractall(destination)


def download_and_extract() -> Path:
    """返回解压后的对象目录；重复运行不会重复下载。"""
    if COMPLETION_MARKER.is_file() and OBJECTS_DIR.is_dir():
        print(f"对象资源已存在，跳过下载: {OBJECTS_DIR}")
        return OBJECTS_DIR

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "缺少 huggingface_hub，请先运行: "
            "python -m pip install -r requirements-assets.txt"
        ) from error

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 {REPOSITORY_ID}/objects.zip 到 {ASSETS_DIR}")
    snapshot_download(
        repo_id=REPOSITORY_ID,
        allow_patterns=["objects.zip"],
        local_dir=str(ASSETS_DIR),
        repo_type="dataset",
    )

    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"下载完成但找不到压缩包: {ARCHIVE_PATH}")

    print(f"解压到 {ASSETS_DIR}")
    _safe_extract(ARCHIVE_PATH, ASSETS_DIR)
    if not OBJECTS_DIR.is_dir() or not any(OBJECTS_DIR.iterdir()):
        raise RuntimeError(f"解压完成但没有找到对象目录: {OBJECTS_DIR}")

    COMPLETION_MARKER.write_text(
        f"source={REPOSITORY_ID}\narchive={ARCHIVE_PATH.name}\n",
        encoding="utf-8",
    )
    ARCHIVE_PATH.unlink()
    print(f"对象资源准备完成: {OBJECTS_DIR}")
    return OBJECTS_DIR


if __name__ == "__main__":
    download_and_extract()

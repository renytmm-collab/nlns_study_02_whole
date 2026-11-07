"""Build an upload archive from an explicit source allowlist; excludes local assets/runs."""
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist"
    output.mkdir(exist_ok=True)
    target = output / "frozen_repair_precheck_v1.zip"
    files = [root / name for name in ("README.md", "LICENSE", "requirements.txt", ".gitignore", "vendor/UPSTREAM.json",
                                      "configs/experiment.example.json")]
    for folder in ("frozen_repair", "tests", "scripts", "docs", "vendor/NLNS"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    hashes = {}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative = path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, "frozen_repair_precheck/"+relative)
        archive.writestr("frozen_repair_precheck/PACKAGE_MANIFEST.json", json.dumps(hashes, indent=2)+"\n")
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError("Archive checksum failed: "+bad)
    print(json.dumps(dict(path=str(target), files=len(files), bytes=target.stat().st_size,
                          sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()

import platform
import subprocess
import sys
from .common import ROOT, COMMIT, file_hash, read_json, Blocked


def source_identity():
    provenance = read_json(ROOT / "vendor/UPSTREAM.json")
    if provenance["commit"] != COMMIT:
        raise Blocked("Upstream provenance commit mismatch")
    for name, expected in provenance["files"].items():
        if file_hash(ROOT / "vendor/NLNS" / name) != expected:
            raise Blocked(f"Unrecorded upstream source modification: {name}")
    paths = []
    for folder in ("frozen_repair", "tests", "scripts", "vendor/NLNS"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and
                     p.suffix in (".py", ".sh", ".json") and "__pycache__" not in p.parts)
    paths.extend([ROOT / "requirements.txt", ROOT / "vendor/UPSTREAM.json"])
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in sorted(paths)}


def environment():
    out = dict(python=sys.version, platform=platform.platform(), device="cpu")
    for name in ("numpy", "torch"):
        try:
            mod = __import__(name)
            out[name] = mod.__version__
        except ImportError:
            out[name] = None
    return out


def git_state():
    def get(*args):
        p = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args],
                           capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return dict(commit=get("rev-parse", "HEAD"), status=get("status", "--porcelain"),
                upstream_commit=COMMIT, upstream_url="https://github.com/ahottung/NLNS")

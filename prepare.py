"""Download the public corpus with revision and checksum; split BEFORE windowing."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent


def prepare():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    target = directory / "input.txt"
    if not target.exists():
        response = requests.get("https://api.github.com/repos/karpathy/char-rnn/commits", params={"path": "data/tinyshakespeare/input.txt", "per_page": 1}, timeout=30)
        response.raise_for_status()
        revision = response.json()[0]["sha"]
        source = f"https://raw.githubusercontent.com/karpathy/char-rnn/{revision}/data/tinyshakespeare/input.txt"
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
        (directory / "source.json").write_text(json.dumps({"url": source, "revision": revision, "corpus": "Tiny Shakespeare", "author": "William Shakespeare", "source_repository": "karpathy/char-rnn", "text": "public-domain Shakespeare excerpts"}, indent=2))
    raw = target.read_bytes()
    n = len(raw)
    boundaries = {"train": [0, int(n * .8)], "validation": [int(n * .8), int(n * .9)], "test": [int(n * .9), n]}
    manifest = json.loads((directory / "source.json").read_text())
    manifest.update({"sha256": hashlib.sha256(raw).hexdigest(), "total_bytes": n, "splits": boundaries, "strategy": "contiguous 80/10/10 byte offsets; no window crosses a split"})
    for split, (left, right) in boundaries.items():
        (directory / f"{split}.bin").write_bytes(raw[left:right])
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()

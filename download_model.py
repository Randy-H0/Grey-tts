"""Downloads Grey models from thisvoiceisai.com into models/: resumes an interrupted download, checks each file's
SHA-256 before it's used. The list (with what each model can do): https://thisvoiceisai.com/studio-models/catalog.json

    python download_model.py v16plus        python download_model.py --list
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CATALOG = "https://thisvoiceisai.com/studio-models/catalog.json"
DEST = Path(__file__).resolve().parent / "models"
UA = {"User-Agent": "Grey-CLI/1.0"}


def _get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=60)


def catalog():
    with _get(CATALOG) as r:
        return json.loads(r.read().decode("utf-8"))["models"]


def fetch(f):
    dst = DEST / f.get("path", f["name"])
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == f["size"]:
        print(f"  {f['name']}: already here")
        return
    have = part.stat().st_size if part.exists() else 0
    with _get(f["url"], {"Range": f"bytes={have}-"} if have else None) as r:
        if have and r.status != 206:
            have = 0
        with open(part, "ab" if have else "wb") as out:
            got = have
            while chunk := r.read(1 << 20):
                out.write(chunk)
                got += len(chunk)
                print(f"\r  {f['name']}: {got / 1e6:.0f} / {f['size'] / 1e6:.0f} MB", end="", flush=True)
    print()
    h = hashlib.sha256()
    with open(part, "rb") as fh:
        while b := fh.read(1 << 22):
            h.update(b)
    if part.stat().st_size != f["size"] or h.hexdigest() != f["sha256"]:
        part.unlink()
        sys.exit(f"  {f['name']}: the download didn't match its fingerprint and was thrown away: try again")
    part.replace(dst)


def main(name="v16plus", list_only=False):
    models = catalog()
    if list_only:
        for m in [x for x in models if not x.get("hidden")]:
            c = m.get("caps", {})
            size = sum(f["size"] for f in m["files"]) / 1e6
            print(f"{m['id']:10s} {m['name']:18s} {size:6.0f} MB  languages: {' '.join(c.get('langs', []))}"
                  f"{'  +own voices' if c.get('clone') else ''}{'  +voice to voice' if c.get('vtv') else ''}")
        return
    m = next((x for x in models if x["id"] == name), None)
    if m is None:
        sys.exit(f"No model called {name!r}: python download_model.py --list")
    DEST.mkdir(exist_ok=True)
    print(f"{m['name']} -> {DEST}")
    for f in m["files"]:
        fetch(f)
    helpers = next((x for x in models if x["id"] == "helpers"), None)
    if helpers and m["id"] != "helpers":  # (voice to voice and fast decoding need them; fetched once)
        print("Helper files (voice to voice, fast decoding):")
        for f in helpers["files"]:
            fetch(f)
    print("done")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "v16plus", "--list" in sys.argv)

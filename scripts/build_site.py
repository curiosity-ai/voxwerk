"""Builds the static voice browser (GitHub Pages) from the voice packages.

Usage: python scripts/build_site.py [out_dir]   (default: _site)

Copies site/ and writes <out>/voices.json (the Voxwerk.Voices.English catalog, with a "sample" path for every voice
whose reference sample exists in an audio package) plus <out>/samples/<id>.opus.
"""
import json, os, shutil, sys

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "_site"))
voices_dir = os.path.join(repo, "src", "Voices")

if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(os.path.join(repo, "site"), out)
os.makedirs(os.path.join(out, "samples"))

catalog = json.load(open(os.path.join(voices_dir, "Voxwerk.Voices.English", "Resources", "voices.json"), encoding="utf-8"))
with_sample = 0
for v in catalog:
    src = os.path.join(voices_dir, v.get("sample_package", ""), "Resources", "samples", v["id"] + ".opus")
    if v.get("sample_package") and os.path.isfile(src):
        shutil.copyfile(src, os.path.join(out, "samples", v["id"] + ".opus"))
        v["sample"] = f"samples/{v['id']}.opus"
        with_sample += 1
json.dump(catalog, open(os.path.join(out, "voices.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
open(os.path.join(out, ".nojekyll"), "w").close()
print(f"{len(catalog)} voices ({with_sample} with samples) -> {out}")

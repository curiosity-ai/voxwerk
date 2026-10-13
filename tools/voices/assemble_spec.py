"""Assembles the voice-builder spec from the rewritten metadata batches and checks originality.

Usage: python assemble_spec.py <meta_in_dir> <meta_out_dir> <spec_out.json> [<problems_out.json>]

* ids: <lang>-<country>-<first>-<last> in kebab-case (ASCII-folded), unique across all packages;
* names must be unique and must not reuse any original catalog name;
* descriptions / scripts must not copy any 4-word phrase from the original catalog text or preview transcript;
* the spoken script is "Hi, I'm <First>. <intro> <sentence>".
Entries that fail a check are written to the problems file for rewriting and left out of the spec.
"""
import glob, hashlib, json, os, re, sys, unicodedata

meta_in, meta_out, spec_path = sys.argv[1], sys.argv[2], sys.argv[3]
problems_path = sys.argv[4] if len(sys.argv) > 4 else None

def fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def kebab(s):
    return re.sub(r"[^a-z0-9]+", "-", fold(s).lower()).strip("-")

def words(s):
    return re.findall(r"[a-z0-9']+", fold(s).lower())

def grams(s, n=4):
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

inputs, originals = {}, set()
for f in sorted(glob.glob(os.path.join(meta_in, "batch_*.json"))):
    for e in json.load(open(f, encoding="utf-8")):
        inputs[e["source_id"]] = e
        if e["original_name"]: originals.add(kebab(e["original_name"]))
# common filler 4-grams that are fine to share
allowed = {"perfect for a walk", "in the park and", "one of the most"}

spec, problems, seen_names, seen_ids = [], [], set(), set()
for f in sorted(glob.glob(os.path.join(meta_out, "batch_*.json"))):
    if ".part" in f: continue
    for o in json.load(open(f, encoding="utf-8")):
        e = inputs.get(o.get("source_id"))
        if e is None: continue
        issues = []
        name = " ".join(o["name"].split())
        first = name.split()[0]
        nk = kebab(name)
        if len(name.split()) < 2: issues.append("name needs first and last name")
        if nk in originals or kebab(e["original_name"]) in nk or nk in kebab(e["original_name"]):
            issues.append("name reuses an original catalog name")
        if nk in seen_names: issues.append("duplicate name")
        script = f"Hi, I'm {first}. {o['intro'].strip()} {o['sentence'].strip()}"
        src = grams(e["original_text"]) | grams(e["transcript"])
        copied = (grams(o["description"]) | grams(o["intro"]) | grams(o["sentence"]) | grams(o["design_prompt"])) & src - allowed
        if copied: issues.append("copied phrases: " + "; ".join(sorted(copied)[:3]))
        if o.get("gender") not in ("female", "male"): issues.append("bad gender")
        if o.get("age") not in ("young", "middle-aged", "old"): issues.append("bad age")
        vid = f"en-{e['country']}-{nk}"
        if vid in seen_ids: issues.append("duplicate id")
        if issues:
            problems.append({"source_id": e["source_id"], "batch": os.path.basename(f), "name": name, "issues": issues})
            continue
        seen_names.add(nk); seen_ids.add(vid)
        spec.append({
            "id": vid, "name": name, "language": "en", "country": e["country"], "region": e["region"],
            "category": e["category"], "gender": o["gender"], "age": o["age"],
            "description": " ".join(o["description"].split()), "tags": [t.lower() for t in o["tags"]][:6],
            "design_prompt": " ".join(o["design_prompt"].split()), "script": " ".join(script.split()),
            "seed": int(hashlib.sha1(vid.encode()).hexdigest()[:6], 16), "source_id": e["source_id"],
        })
json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if problems_path: json.dump(problems, open(problems_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(spec)} voices in spec, {len(problems)} with problems")
for p in problems[:10]: print("  ", p["batch"], p["name"], p["issues"])

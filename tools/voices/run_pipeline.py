"""Runs the voice build end to end, resumably (see README "Voice packages").

Usage: python run_pipeline.py --work /tmp/voxwerk-build/work --design-model <1.7B-VoiceDesign dir>
                              --base-model <0.6B-Base dir> --python <python with qwen-tts> [--batch 24] [--threads 4]

Each round: assemble spec from rewritten metadata -> VoiceDesign generation -> decode -> QA -> retry failures with new
seeds (up to 3 attempts) -> Base-model cloning of passing voices -> decode -> QA -> pack into src/Voices.
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--design-model", required=True)
ap.add_argument("--base-model", required=True)
ap.add_argument("--python", required=True)
ap.add_argument("--meta-in", default="/tmp/voxwerk-build/meta_in")
ap.add_argument("--meta-out", default="/tmp/voxwerk-build/meta_out")
ap.add_argument("--batch", default="24")
ap.add_argument("--threads", default="4")
ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ap.add_argument("--max-attempts", type=int, default=3)
a = ap.parse_args()

repo, work = a.repo, a.work
os.makedirs(work, exist_ok=True)
spec = os.path.join(work, "spec.json")
builder = ["dotnet", os.path.join(repo, "tools/Voxwerk.VoiceBuilder/bin/Release/net10.0/Voxwerk.VoiceBuilder.dll")]
attempts_path = os.path.join(work, "attempts.json")
attempts = json.load(open(attempts_path)) if os.path.exists(attempts_path) else {}

def run(cmd, **kw):
    print(time.strftime("%H:%M:%S"), " ".join(cmd[:4]), "...", flush=True)
    subprocess.run(cmd, check=True, **kw)

def qa():
    run([a.python, "-I", os.path.join(repo, "tools/voices/qa.py"), spec, work, a.base_model, "--threads", a.threads], cwd="/tmp")
    return json.load(open(os.path.join(work, "qa.json")))

run([a.python if False else sys.executable, os.path.join(repo, "tools/voices/assemble_spec.py"), a.meta_in, a.meta_out, spec,
     os.path.join(work, "problems.json")])
common = ["--spec", spec, "--work", work, "--batch", a.batch, "--threads", a.threads]
run(builder + ["generate", "--model", a.design_model] + common)
run(builder + ["decode", "--model", a.design_model, "--work", work, "--threads", a.threads])
results = qa()

# Retry failing designs with new seeds.
for attempt in range(1, a.max_attempts):
    failed = [k for k, v in results.items() if not v["pass"] and not k.endswith(".clone") and attempts.get(k, 0) < attempt]
    if not failed: break
    for k in failed:
        attempts[k] = attempt
        for p in (os.path.join(work, "codes", k + ".codes"), os.path.join(work, "wav", k + ".wav")):
            if os.path.exists(p): os.remove(p)
        results.pop(k, None)
    json.dump(results, open(os.path.join(work, "qa.json"), "w"), indent=1)
    json.dump(attempts, open(attempts_path, "w"), indent=1)
    ids = os.path.join(work, f"retry{attempt}.txt")
    open(ids, "w").write("\n".join(failed))
    run(builder + ["generate", "--model", a.design_model, "--only", ids, "--seed-offset", str(1000 * attempt)] + common)
    run(builder + ["decode", "--model", a.design_model, "--work", work, "--threads", a.threads])
    results = qa()

# Clone the passing designs (QA rejects are excluded by removing their wavs from the clone input list).
passing = [k for k, v in results.items() if v["pass"] and not k.endswith(".clone")]
ids = os.path.join(work, "clone_ids.txt")
open(ids, "w").write("\n".join(passing))
run(builder + ["clone", "--model", a.base_model, "--only", ids] + common)
run(builder + ["decode", "--model", a.base_model, "--work", work, "--threads", a.threads])
results = qa()
# A clone that fails QA gets one more try with another seed.
failed = [k[:-6] for k, v in results.items() if not v["pass"] and k.endswith(".clone") and attempts.get(k, 0) < 1]
if failed:
    for vid in failed:
        attempts[vid + ".clone"] = 1
        for p in (os.path.join(work, "codes", vid + ".clone.codes"), os.path.join(work, "wav", vid + ".clone.wav")):
            if os.path.exists(p): os.remove(p)
        results.pop(vid + ".clone", None)
    json.dump(results, open(os.path.join(work, "qa.json"), "w"), indent=1)
    json.dump(attempts, open(attempts_path, "w"), indent=1)
    open(ids, "w").write("\n".join(failed))
    run(builder + ["clone", "--model", a.base_model, "--only", ids, "--seed-offset", "500"] + common)
    run(builder + ["decode", "--model", a.base_model, "--work", work, "--threads", a.threads])
    results = qa()

run(builder + ["pack", "--spec", spec, "--work", work, "--qa", os.path.join(work, "qa.json"), "--out", os.path.join(repo, "src/Voices")])
ok = sum(1 for k, v in results.items() if v["pass"] and not k.endswith(".clone"))
print(f"done: {ok} voices pass QA of {len(json.load(open(spec)))} in spec", flush=True)

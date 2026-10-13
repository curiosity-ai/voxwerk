"""Automatic quality gate for generated voice samples (tools/Voxwerk.VoiceBuilder output).

Usage: python qa.py <spec.json> <work_dir> <base_model_dir> [--threads 4]

For every <work>/wav/<id>.wav (design) and <id>.clone.wav (clone) it checks:
  * intelligibility: Whisper word error rate against the intended text (tiny.en first, base.en re-check),
  * gender plausibility: median F0 vs. the specified gender,
  * pacing: characters per second within a sane range (catches truncation / run-on generations),
  * clone fidelity: speaker-embedding cosine similarity between the clone and its design sample.
Results (merged into <work>/qa.json) feed `voicebuilder pack --qa` and regeneration with a new seed.
"""
import json, os, re, sys
import numpy as np, soundfile as sf, librosa, torch
from transformers import pipeline

CLONE_SENTENCE = "Every voice in this collection reads this same sentence, so you can compare how each one sounds side by side."
spec_path, work, base_dir = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4)
spec = {s["id"]: s for s in json.load(open(spec_path, encoding="utf-8"))}
qa_path = os.path.join(work, "qa.json")
qa = json.load(open(qa_path)) if os.path.exists(qa_path) else {}

def norm(t):
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

asr_tiny = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
asr_base = None
enc = None

def speaker_model():
    global enc
    if enc is None:
        from qwen_tts import Qwen3TTSModel
        enc = Qwen3TTSModel.from_pretrained(base_dir, device_map="cpu", dtype=torch.float32).model
    return enc

files = sorted(f for f in os.listdir(os.path.join(work, "wav")) if f.endswith(".wav"))
n = 0
for f in files:
    key = f[:-4]
    if key in qa: continue
    clone = key.endswith(".clone")
    vid = key[:-6] if clone else key
    s = spec.get(vid)
    if s is None: continue
    a, sr = sf.read(os.path.join(work, "wav", f), dtype="float32")
    dur = len(a) / sr
    text = CLONE_SENTENCE if clone else s["script"]
    a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
    first = s["name"].split()[0].lower()
    ref_words = " ".join(w for w in norm(text) if w != first)
    hyp = asr_tiny(a16)["text"]
    hyp_words = " ".join(w for w in norm(hyp) if w != first)
    w = wer(ref_words, hyp_words)
    if w > 0.1:  # re-check with a stronger model before blaming the voice
        if asr_base is None:
            asr_base = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
        hyp = asr_base(a16)["text"]
        w = min(w, wer(ref_words, " ".join(x for x in norm(hyp) if x != first)))
    f0 = librosa.yin(a, fmin=60, fmax=450, sr=sr)
    voiced = f0[(f0 > 65) & (f0 < 440)]
    med = float(np.median(voiced)) if len(voiced) else 0.0
    cps = len(text) / max(dur, 0.1)
    reasons = []
    if w > 0.2: reasons.append(f"wer {w:.2f}")
    if s["gender"] == "male" and med > 185: reasons.append(f"f0 {med:.0f} Hz for male")
    if s["gender"] == "female" and med < 150: reasons.append(f"f0 {med:.0f} Hz for female")
    if not 6 <= cps <= 25: reasons.append(f"pace {cps:.1f} chars/s")
    entry = {"pass": not reasons, "wer": round(w, 3), "f0": round(med), "seconds": round(dur, 2), "cps": round(cps, 1), "asr": hyp.strip(), "reasons": reasons}
    if clone:
        design = os.path.join(work, "wav", vid + ".wav")
        if os.path.exists(design):
            m = speaker_model()
            d, _ = sf.read(design, dtype="float32")
            sim = torch.nn.functional.cosine_similarity(m.extract_speaker_embedding(a, 24000), m.extract_speaker_embedding(d, 24000), dim=0).item()
            entry["speaker_sim"] = round(sim, 3)
            if sim < 0.75:
                entry["pass"] = False
                entry["reasons"].append(f"speaker sim {sim:.2f}")
    qa[key] = entry
    n += 1
    if n % 50 == 0:
        json.dump(qa, open(qa_path, "w"), indent=1)
        print(f"{n} checked, {sum(1 for v in qa.values() if not v['pass'])} failing so far", flush=True)
json.dump(qa, open(qa_path, "w"), indent=1)
fails = {k: v for k, v in qa.items() if not v["pass"]}
print(f"checked {n}; total {len(qa)}, failing {len(fails)}")
for k, v in list(fails.items())[:20]: print("  FAIL", k, v["reasons"], repr(v["asr"][:80]))

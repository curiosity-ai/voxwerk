"""Times the official qwen-tts PyTorch implementation phase by phase (for comparison with `voxwerk synth`).

Usage:
    python bench_reference.py <model_dir> <ref_wav> --ref-text "..." --text "..." [--threads 1] [--greedy]
                              [--max-frames N] [--seed S] [--out out.wav] [--dtype float32|bfloat16]
    python bench_reference.py <voice_design_dir> --mode design --voice "..." --text "..." [...]
    python bench_reference.py <custom_voice_dir> --mode custom --speaker ryan [--instruct "..."] --text "..." [...]

Reports the same breakdown the C# CLI prints: voice-prompt time (speaker encoder + speech tokenizer encoder),
talker time (prefill + autoregressive generation, incl. the code predictor) and speech-decoder time.
"""
import argparse
import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("ref_wav", nargs="?")
ap.add_argument("--mode", default="clone", choices=["clone", "design", "custom"])
ap.add_argument("--ref-text")
ap.add_argument("--voice")
ap.add_argument("--speaker")
ap.add_argument("--instruct")
ap.add_argument("--text", required=True)
ap.add_argument("--language", default="English")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--max-frames", type=int, default=2048)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--out", default=None)
a = ap.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(a.model_dir, device_map="cpu", dtype=getattr(torch, a.dtype))
print(f"load {time.time() - t0:.1f}s, threads={torch.get_num_threads()}, dtype={a.dtype}")

timings = {}


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t = time.time()
        r = fn(*args, **kwargs)
        timings[name] = timings.get(name, 0.0) + time.time() - t
        return r
    return wrapper


tts.model.generate = timed("talker", tts.model.generate)
tts.model.speech_tokenizer.decode = timed("decoder", tts.model.speech_tokenizer.decode)

t = time.time()
t_prompt = 0.0
common = dict(text=a.text, language=a.language, do_sample=not a.greedy, subtalker_dosample=not a.greedy, max_new_tokens=a.max_frames)
if a.mode == "clone":
    prompt = tts.create_voice_clone_prompt(ref_audio=a.ref_wav, ref_text=a.ref_text)
    t_prompt = time.time() - t
    t = time.time()
    wavs, sr = tts.generate_voice_clone(voice_clone_prompt=prompt, **common)
elif a.mode == "design":
    wavs, sr = tts.generate_voice_design(instruct=a.voice, **common)
else:
    wavs, sr = tts.generate_custom_voice(speaker=a.speaker, instruct=a.instruct, **common)
total = time.time() - t + t_prompt
audio = len(wavs[0]) / sr
frames = round(audio * 12.5)
print(f"audio {audio:.2f}s ({frames} frames)")
print(f"  prompt {t_prompt:.2f}s | talker {timings['talker']:.2f}s ({timings['talker'] * 1000 / max(frames, 1):.0f} ms/frame)"
      f" | decoder {timings['decoder']:.2f}s")
print(f"  total {total:.2f}s, RTF {total / audio:.3f}")
if a.out:
    sf.write(a.out, wavs[0], sr)

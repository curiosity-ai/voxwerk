# voxwerk

Pure C# (.NET 10) inference for **[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)**,
Qwen's voice-cloning text-to-speech model, as a library plus a command-line tool.

- No Python, no ONNX, no native dependencies: tokenizer, transformers, codec and vocoder are all managed
  code with hand-written AVX-512 / AVX2 kernels (`System.Runtime.Intrinsics`). [NAudio](https://github.com/naudio/NAudio)
  is used for WAV I/O and resampling.
- Validated layer by layer against the official PyTorch implementation (`qwen-tts`), see [Validation](#validation).
- About **2.2–2.5× faster than the PyTorch reference on a single CPU core**, see [Performance](#performance).

## Quick start

```bash
# Synthesize in a cloned voice (the checkpoint is downloaded to ~/.cache/voxwerk on first use)
dotnet run -c Release --project src/Voxwerk.Cli -- synth \
    --ref reference.wav \
    --ref-text "Transcript of reference.wav" \
    --text "Hello! This voice was cloned from a few seconds of audio." \
    --language english \
    --out hello.wav
```

`voxwerk synth` options (run `voxwerk --help` for all of them):

| Option | Meaning |
|---|---|
| `--ref <wav>` | Reference recording to clone (any sample rate or channel count; mixed to mono and resampled to 24 kHz) |
| `--ref-text <text>` | Transcript of the reference. Enables ICL mode (best similarity). Without it, x-vector-only cloning is used |
| `--x-vector-only` | Clone from the speaker embedding only |
| `--language <name>` | `auto` (default), `english`, `chinese`, `german`, `french`, `spanish`, `italian`, `portuguese`, `russian`, `japanese`, `korean` |
| `--model <dir\|repo>` | Local model directory or Hugging Face repo id (default `Qwen/Qwen3-TTS-12Hz-0.6B-Base`) |
| `--threads <n>` | Worker threads (default: all cores; `1` = single core) |
| `--int8` | Store talker weights as int8 (faster, approximate; see below) |
| `--seed`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--sub-temperature`, `--sub-top-k`, `--max-frames`, `--greedy`, `--non-streaming` | Generation settings, with the same defaults as the model's `generation_config.json` |

### Library

```csharp
using Voxwerk;
using Voxwerk.Audio;
using Voxwerk.Talker;

using var tts = await Qwen3Tts.LoadAsync();                     // or Qwen3Tts.Load("/path/to/model", threads: 1)
var voice = tts.CreateVoiceClonePrompt("reference.wav", "Transcript of reference.wav");
var result = tts.Synthesize("Hello from C#!", voice, language: "english",
                            new GenerationOptions { Seed = 42 });
AudioFile.WriteWav("hello.wav", result.Audio, result.SampleRate);
Console.WriteLine($"{result.AudioSeconds:F1}s of audio, RTF {result.RealTimeFactor:F2}");
```

A `VoiceClonePrompt` (speaker x-vector + reference codes) can be built once and reused for many sentences.
Lower-level pieces (`GenerateCodes`, `Decode`, `Qwen2Tokenizer`, `SpeechEncoder`, `SpeechDecoder`,
`SpeakerEncoder`) are public as well.

## How it works

```
reference.wav ──► log-mel (STFT, Slaney mel) ──► ECAPA-TDNN speaker encoder ──► x-vector ─────────┐
              └─► Mimi encoder (SEANet + transformer + split RVQ) ──► reference codes (12.5 Hz × 16) ┤
text ──► Qwen2 BPE ──► text embedding + projection ──────────────────────────────────────────────────┤
                                                                                                     ▼
            talker: 28-layer Qwen3 decoder ──► codebook 0 ──► code predictor (5 layers) ──► codebooks 1..15
                                                                                                     ▼
            speech decoder: RVQ ► causal conv ► 8-layer sliding-window transformer ► 2×(ConvT + ConvNeXt)
                            ► SnakeBeta vocoder (×8·5·4·3) ──► 24 kHz waveform (1920 samples per frame)
```

| Source | Contents |
|---|---|
| `src/Voxwerk/Qwen3Tts.cs` | Public API: load, voice prompt, synthesize |
| `src/Voxwerk/Talker/` | Config, Qwen3 decoder stack with KV cache, talker + code predictor, prompt construction and sampling loop |
| `src/Voxwerk/Codec/` | 12 Hz speech tokenizer: Mimi-style encoder, decoder/vocoder, codec transformers |
| `src/Voxwerk/Speaker/`, `src/Voxwerk/Audio/` | ECAPA-TDNN speaker encoder, mel spectrogram, NAudio-based audio I/O |
| `src/Voxwerk/Text/` | Qwen2 byte-level BPE tokenizer (`vocab.json` + `merges.txt`) |
| `src/Voxwerk/Numerics/` | GEMM/GEMV kernels, convolutions, attention, RoPE, norms, activations |
| `src/Voxwerk/IO/` | safetensors reader, Hugging Face downloader |
| `src/Voxwerk.Cli/` | `voxwerk` command-line tool |
| `tests/Voxwerk.Tests/` | Kernel unit tests and layer-by-layer parity tests |
| `scripts/` | Reference tensor dump and PyTorch benchmark scripts |

### Performance design

Autoregressive decoding of a 12.5 Hz frame runs the talker once and the code predictor 15 times, i.e.
~1.3 B multiply-adds over **3.2 GB of weights per frame in bf16** (6.4 GB in fp32). On one core that is purely
memory-bandwidth bound (this machine streams ~9.5 GB/s per core), so the design minimizes bytes per frame:

- **Weights stay in bf16**, the checkpoint's native precision, packed once at load into 32-column panels that
  the GEMV streams exactly once per token and widens in-register (lossless). PyTorch on CPU runs this model in
  fp32 and moves twice the bytes.
- q/k/v and gate/up projections are fused, the KV cache is head-major and each KV head is read once for the
  query heads that share it (GQA).
- Optional **int8** talker weights (per-output-channel scales) halve the bytes again.

The speech decoder is compute bound (~58 GFLOP per second of audio):

- Convolutions are GEMMs over channels-last activations against pre-packed weights. Undilated windows are read
  as overlapping rows straight from the padded input (no im2col), and transposed convolutions become a single
  GEMM with no overlap-add pass. Narrow layers (96–768 channels) use a 4×96 register-tile kernel that reads
  dilated taps by pointer offset.
- SnakeBeta is fused into the conv input preparation, and the residual add plus bias into the GEMM epilogue.
  `sin`/`exp` are custom vectorized polynomials (~5× faster than `Vector512.Sin`).
- The audio-rate vocoder runs **streamed in causal segments**, carrying each conv's left context. Results are
  bit-identical to whole-sequence processing (checked by a test), but the working set stays cache-sized instead
  of streaming gigabytes of activations through memory. Large buffers are pooled and backed by huge pages.

## Validation

`scripts/dump_reference.py` runs the official `qwen-tts` package with greedy decoding and saves every
intermediate tensor; the xUnit parity tests compare the C# implementation against them:

| Component | Checked | Result (relative L2 error unless noted) |
|---|---|---|
| Tokenizer | ids for chat prompts, CJK, accents, emoji, whitespace runs | identical |
| Log-mel front end | 128-bin mel of the reference | 3.1e-7 |
| Speaker encoder | each ECAPA block, MFA, pooling, x-vector | ≤ 1.3e-6 (x-vector 9.2e-7) |
| Speech tokenizer encoder | all 15 SEANet layers, transformer, downsample | ≤ 1.5e-5; **codes identical (0 / 1616)** |
| Talker prompt | ICL and x-vector prefill embeddings | 3.1e-7 |
| Talker | all 28 layer outputs on a 111-token prefill | ≤ 1.7e-5 |
| Talker / code predictor | logits per decode step | ≤ 4.1e-6 |
| Generation | greedy codec frames, ICL and x-vector modes | **identical (0 / 368)** |
| Speech decoder | RVQ, pre-conv, transformer, upsamplers, every vocoder block | ≤ 6.2e-6 |
| Waveform | 9.9 s decoded audio | 2.7e-6 (max abs 5.1e-6) |

End-to-end checks with sampling (default settings, seed 1234, single thread). Whisper transcription, plus
cosine similarity to the reference voice measured with the reference implementation's own speaker encoder:

| Output | Speaker similarity | Whisper (base.en) transcript |
|---|---|---|
| PyTorch reference | 0.991 | "Hello there. This is a quick test of the text to speech system running on a single CPU core." |
| Voxwerk (bf16) | 0.990 | "Hello there! This is a quick test of the text-to-speech system running on a single CPU core." |
| Voxwerk (`--int8`) | 0.989 | "Hello there. This is a quick test of the text to speech system running on a single CPU core." |

Note: pure greedy decoding degenerates to near-silence for this model in both implementations (that is why the
reference samples by default). It is only used to make parity tests deterministic.

### Running the tests

```bash
dotnet test -c Release                       # kernel unit tests (parity tests skip without fixtures)

# Parity tests: dump reference tensors with the official implementation first
python -m venv .venv && .venv/bin/pip install qwen-tts torch torchaudio
.venv/bin/python scripts/dump_reference.py /path/to/Qwen3-TTS-12Hz-0.6B-Base reference.wav /tmp/refdump
VOXWERK_MODEL_DIR=/path/to/Qwen3-TTS-12Hz-0.6B-Base VOXWERK_REFERENCE_DIR=/tmp/refdump dotnet test -c Release
```

The reference voice used here is the sample from the model card
(`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav`).

## Performance

Single core (`--threads 1` vs `torch.set_num_threads(1)`), Intel Xeon @ 2.8 GHz (AVX-512) cloud VM, .NET 10,
PyTorch 2.11 (CPU, fp32: the reference's CPU dtype), `qwen-tts` 0.1.1. ICL voice clone from the 8 s model-card
sample. The VM's timings vary by about ±10% between runs.

**Same workload** (greedy, capped at 100 frames, so both produce exactly 99 frames / 7.9 s of audio):

| Phase | PyTorch | Voxwerk (bf16) | Speed-up |
|---|---|---|---|
| Voice prompt (speaker + speech encoder) | 3.2 s | 3.2 s | 1.0× |
| Talker + code predictor | 95.3 s (962 ms/frame) | 41.3 s (417 ms/frame) | 2.3× |
| Speech decoder (99 frames + 101 reference-context frames) | 36.8 s | 13.6 s | 2.7× |
| **Total** | **135.2 s (RTF 17.1)** | **58.2 s (RTF 7.3)** | **2.3×** |

**Typical sampled run** (default sampling, natural end of speech):

| | Audio | Wall time | RTF |
|---|---|---|---|
| PyTorch | 5.84 s | 111.0 s | 19.0 |
| Voxwerk (bf16) | 5.44 s | 48.7 s | 8.9 |
| Voxwerk (`--int8`) | 5.52 s | 35.1 s | 6.4 |

Reproduce with `scripts/bench_reference.py` (prints the same phase breakdown as `voxwerk synth`):

```bash
python scripts/bench_reference.py /path/to/model reference.wav --ref-text "..." --text "..." --threads 1 --greedy --max-frames 100
voxwerk synth --model /path/to/model --ref reference.wav --ref-text "..." --text "..." --threads 1 --greedy --max-frames 100
```

Multi-threading (`--threads N`, default all cores) parallelizes the GEMMs, convolutions and attention. The same
fixed workload on 4 threads: PyTorch 62.7 s (RTF 7.9; talker 447 ms/frame, decoder 15.8 s) vs Voxwerk 35.2 s
(RTF 4.4; talker 249 ms/frame, decoder 8.0 s).

## Building

```bash
dotnet build -c Release
```

Requires the .NET 10 SDK. AVX-512 and AVX2+FMA are used when available; other CPUs fall back to portable
(slower) code paths.

## Model license

The Qwen3-TTS weights (downloaded separately from Hugging Face) are released by the Qwen team under Apache-2.0.

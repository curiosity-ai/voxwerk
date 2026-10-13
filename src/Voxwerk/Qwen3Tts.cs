using System.Diagnostics;
using Voxwerk.Audio;
using Voxwerk.Codec;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Voxwerk.Speaker;
using Voxwerk.Talker;
using Voxwerk.Text;

namespace Voxwerk;

/// <summary>Result of a synthesis call.</summary>
public sealed record SynthesisResult(float[] Audio, int SampleRate, int[][] Codes, TimeSpan PromptTime, TimeSpan TalkerTime, TimeSpan DecoderTime)
{
    public double AudioSeconds => (double)Audio.Length / SampleRate;
    public TimeSpan TotalTime => PromptTime + TalkerTime + DecoderTime;
    /// <summary>Real-time factor: processing time / audio duration (lower is faster).</summary>
    public double RealTimeFactor => AudioSeconds > 0 ? TotalTime.TotalSeconds / AudioSeconds : double.NaN;
}

/// <summary>
/// Qwen3-TTS 12 Hz 0.6B Base (voice cloning text-to-speech) in pure C#.
/// <code>
/// using var tts = Qwen3Tts.Load(modelDir);
/// var voice = tts.CreateVoiceClonePrompt("ref.wav", "Transcript of ref.wav");
/// var result = tts.Synthesize("Hello world!", voice, "English");
/// AudioFile.WriteWav("out.wav", result.Audio);
/// </code>
/// </summary>
public sealed class Qwen3Tts : IDisposable
{
    public const int SampleRate = 24000;

    private readonly TalkerModel _talker;
    private readonly TalkerGenerator _generator;
    private readonly SpeakerEncoder _speaker;
    private readonly SpeechEncoder _encoder;
    private readonly SpeechDecoder _decoder;
    private readonly MelSpectrogram _mel = new();

    private Qwen3Tts(string dir, Parallelism par)
    {
        Parallelism = par;
        Config = Qwen3TtsConfig.Load(Path.Combine(dir, "config.json"));
        if (!Config.TtsModelType.Equals("base", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"Only the Base (voice clone) model type is supported, got '{Config.TtsModelType}'.");
        Tokenizer = Qwen2Tokenizer.FromDirectory(dir);
        using (var st = new SafeTensorsFile(Path.Combine(dir, "model.safetensors")))
        {
            _talker = new TalkerModel(st, Config);
            _speaker = new SpeakerEncoder(st);
        }
        using (var st = new SafeTensorsFile(Path.Combine(dir, "speech_tokenizer", "model.safetensors")))
        {
            _encoder = new SpeechEncoder(st);
            _decoder = new SpeechDecoder(st);
        }
        _generator = new TalkerGenerator(_talker, Tokenizer);
    }

    /// <summary>Loads a model directory (as downloaded from <c>Qwen/Qwen3-TTS-12Hz-0.6B-Base</c>).</summary>
    /// <param name="threads">Worker threads for the numeric kernels; 1 = single core, ≤ 0 = all cores.</param>
    public static Qwen3Tts Load(string modelDir, int threads = 0) => new(modelDir, new Parallelism(threads));

    /// <summary>Downloads the checkpoint if needed (Hugging Face repo id or local path) and loads it.</summary>
    public static async Task<Qwen3Tts> LoadAsync(string modelOrRepo = ModelDownloader.DefaultRepo, int threads = 0, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        string dir = await ModelDownloader.EnsureAsync(modelOrRepo, progress: progress, ct: ct).ConfigureAwait(false);
        return Load(dir, threads);
    }

    public Qwen3TtsConfig Config { get; }
    public Qwen2Tokenizer Tokenizer { get; }
    public Parallelism Parallelism { get; }
    public IReadOnlyCollection<string> Languages => ["auto", .. Config.LanguageIds.Keys];

    /// <summary>Builds a voice prompt from 24 kHz mono reference audio. Pass the transcript for ICL mode
    /// (best similarity); with <paramref name="xVectorOnly"/> (or no transcript) only the speaker embedding is used.</summary>
    public VoiceClonePrompt CreateVoiceClonePrompt(float[] referenceAudio24k, string? referenceText, bool xVectorOnly = false)
    {
        var mel = _mel.Compute(referenceAudio24k, out int frames);
        var spk = _speaker.Forward(mel, frames, Parallelism);
        bool icl = !xVectorOnly && !string.IsNullOrWhiteSpace(referenceText);
        return new VoiceClonePrompt
        {
            SpeakerEmbedding = spk,
            ReferenceCodes = icl ? _encoder.Encode(referenceAudio24k, Parallelism) : null,
            ReferenceText = icl ? referenceText : null,
        };
    }

    /// <summary>Loads a WAV reference (any rate / channel count; it is mixed down and resampled to 24 kHz).</summary>
    public VoiceClonePrompt CreateVoiceClonePrompt(string referenceWavPath, string? referenceText, bool xVectorOnly = false)
        => CreateVoiceClonePrompt(AudioFile.ReadMono(referenceWavPath, SampleRate), referenceText, xVectorOnly);

    /// <summary>Generates codec frames [frames][16] for <paramref name="text"/>.</summary>
    public int[][] GenerateCodes(string text, VoiceClonePrompt voice, string language = "auto", GenerationOptions? options = null,
                                 CancellationToken ct = default, Action<int>? onFrame = null)
    {
        options ??= new GenerationOptions();
        var prompt = _generator.BuildPrompt(text, language, voice, options.NonStreamingMode, Parallelism);
        return _generator.Generate(prompt, options, Parallelism, ct, onFrame).ToArray();
    }

    /// <summary>Decodes codec frames to 24 kHz audio. When <paramref name="context"/> (reference codes) is given it is
    /// decoded in front as acoustic context and the corresponding audio is cut off, like the reference pipeline.</summary>
    public float[] Decode(IReadOnlyList<int[]> codes, IReadOnlyList<int[]>? context = null, CancellationToken ct = default)
    {
        if (context == null || context.Count == 0) return _decoder.Decode(codes, Parallelism, ct);
        var all = new List<int[]>(context.Count + codes.Count);
        all.AddRange(context);
        all.AddRange(codes);
        var wav = _decoder.Decode(all, Parallelism, ct);
        int cut = (int)((double)context.Count / Math.Max(all.Count, 1) * wav.Length);
        return wav[cut..];
    }

    /// <summary>Text → speech in the cloned voice.</summary>
    public SynthesisResult Synthesize(string text, VoiceClonePrompt voice, string language = "auto", GenerationOptions? options = null,
                                      CancellationToken ct = default, Action<int>? onFrame = null)
    {
        options ??= new GenerationOptions();
        var sw = Stopwatch.StartNew();
        var prompt = _generator.BuildPrompt(text, language, voice, options.NonStreamingMode, Parallelism);
        var tPrompt = sw.Elapsed;
        sw.Restart();
        var codes = _generator.Generate(prompt, options, Parallelism, ct, onFrame).ToArray();
        var tTalker = sw.Elapsed;
        sw.Restart();
        var audio = codes.Length == 0 ? [] : Decode(codes, voice.ReferenceCodes, ct);
        var tDecode = sw.Elapsed;
        return new SynthesisResult(audio, SampleRate, codes, tPrompt, tTalker, tDecode);
    }

    public void Dispose()
    {
        _talker.Dispose();
        _speaker.Dispose();
        _encoder.Dispose();
        _decoder.Dispose();
    }
}

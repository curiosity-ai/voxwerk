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

/// <summary>The three Qwen3-TTS checkpoint families.</summary>
public enum Qwen3TtsModelType
{
    /// <summary><c>*-Base</c>: clones a voice from reference audio (<see cref="Qwen3Tts.Synthesize"/>).</summary>
    Base,

    /// <summary><c>*-VoiceDesign</c>: creates a voice from a natural-language description (<see cref="Qwen3Tts.SynthesizeVoiceDesign"/>).</summary>
    VoiceDesign,

    /// <summary><c>*-CustomVoice</c>: built-in named speakers, optionally steered by an instruction (<see cref="Qwen3Tts.SynthesizeCustomVoice"/>).</summary>
    CustomVoice,
}

/// <summary>
/// Qwen3-TTS (12 Hz tokenizer; 0.6B / 1.7B; Base, VoiceDesign and CustomVoice checkpoints) in pure C#.
/// <code>
/// // Voice cloning (Qwen/Qwen3-TTS-12Hz-0.6B-Base or 1.7B-Base)
/// using var tts = Qwen3Tts.Load(modelDir);
/// var voice = tts.CreateVoiceClonePrompt("ref.wav", "Transcript of ref.wav");
/// var result = tts.Synthesize("Hello world!", voice, "English");
///
/// // Voice design (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
/// var designed = tts.SynthesizeVoiceDesign("Hello world!", "A warm, low female voice, speaking slowly.");
///
/// // Built-in speakers (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice or 1.7B-CustomVoice)
/// var custom = tts.SynthesizeCustomVoice("Hello world!", speaker: "ryan");
/// AudioFile.WriteWav("out.wav", result.Audio);
/// </code>
/// </summary>
public sealed class Qwen3Tts : IDisposable
{
    public const int SampleRate = 24000;

    private readonly TalkerModel _talker;
    private readonly TalkerGenerator _generator;
    private readonly SpeakerEncoder? _speaker;
    private readonly SpeechEncoder? _encoder;
    private readonly SpeechDecoder _decoder;
    private readonly MelSpectrogram _mel = new();

    private Qwen3Tts(string dir, Parallelism par, WeightPrecision precision)
    {
        Parallelism = par;
        Config = Qwen3TtsConfig.Load(Path.Combine(dir, "config.json"));
        ModelType = Config.TtsModelType.ToLowerInvariant() switch
        {
            "base" => Qwen3TtsModelType.Base,
            "voice_design" => Qwen3TtsModelType.VoiceDesign,
            "custom_voice" => Qwen3TtsModelType.CustomVoice,
            var t => throw new NotSupportedException($"Unknown tts_model_type '{t}'."),
        };
        Tokenizer = Qwen2Tokenizer.FromDirectory(dir);
        using (var st = new SafeTensorsFile(Path.Combine(dir, "model.safetensors")))
        {
            _talker = new TalkerModel(st, Config, precision);
            if (ModelType == Qwen3TtsModelType.Base) _speaker = new SpeakerEncoder(st);
        }
        using (var st = new SafeTensorsFile(Path.Combine(dir, "speech_tokenizer", "model.safetensors")))
        {
            // The codec encoder is only needed to turn reference audio into codes (voice cloning).
            if (ModelType == Qwen3TtsModelType.Base) _encoder = new SpeechEncoder(st);
            _decoder = new SpeechDecoder(st);
        }
        _generator = new TalkerGenerator(_talker, Tokenizer);
    }

    /// <summary>Loads a model directory (as downloaded from e.g. <c>Qwen/Qwen3-TTS-12Hz-0.6B-Base</c>).</summary>
    /// <param name="modelDir">Directory with config.json, model.safetensors, tokenizer files and speech_tokenizer/.</param>
    /// <param name="threads">Worker threads for the numeric kernels; 1 = single core, ≤ 0 = all cores.</param>
    /// <param name="precision">Talker weight storage: bf16 (exact, default) or int8 (faster decode steps).</param>
    public static Qwen3Tts Load(string modelDir, int threads = 0, WeightPrecision precision = WeightPrecision.BFloat16)
        => new(modelDir, new Parallelism(threads), precision);

    /// <summary>Downloads the checkpoint if needed (Hugging Face repo id or local path) and loads it.</summary>
    public static async Task<Qwen3Tts> LoadAsync(string modelOrRepo = ModelDownloader.DefaultRepo, int threads = 0,
                                                 WeightPrecision precision = WeightPrecision.BFloat16, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        string dir = await ModelDownloader.EnsureAsync(modelOrRepo, progress: progress, ct: ct).ConfigureAwait(false);
        return Load(dir, threads, precision);
    }

    public Qwen3TtsConfig Config { get; }
    public Qwen3TtsModelType ModelType { get; }
    public Qwen2Tokenizer Tokenizer { get; }
    public Parallelism Parallelism { get; }
    public WeightPrecision Precision => _talker.Precision;

    /// <summary>Lower-level access to prompt construction and (batched) generation.</summary>
    public TalkerGenerator Generator => _generator;

    /// <summary>Accepted <c>language</c> values (dialects are selected through dialect speakers, not directly).</summary>
    public IReadOnlyCollection<string> Languages => ["auto", .. Config.LanguageIds.Keys.Where(k => !k.Contains("dialect"))];

    /// <summary>Built-in speakers of a CustomVoice model (empty otherwise).</summary>
    public IReadOnlyCollection<string> Speakers => Config.SpeakerIds.Keys.ToArray();

    /// <summary>Whether <see cref="SynthesizeCustomVoice"/> honours an instruction. The reference implementation
    /// ignores instructions for the 0.6B CustomVoice checkpoint, and so does this one.</summary>
    public bool SupportsInstruct => ModelType == Qwen3TtsModelType.VoiceDesign ||
                                    (ModelType == Qwen3TtsModelType.CustomVoice && !Config.TtsModelSize.Equals("0b6", StringComparison.OrdinalIgnoreCase));

    private void Require(Qwen3TtsModelType type, string what)
    {
        if (ModelType != type)
            throw new NotSupportedException($"{what} needs a {type} checkpoint (e.g. {ExampleRepo(type)}); the loaded model is {ModelType} ({Config.TtsModelSize}).");
    }

    /// <summary>A Hugging Face repository providing the given model type.</summary>
    public static string ExampleRepo(Qwen3TtsModelType type) => type switch
    {
        Qwen3TtsModelType.VoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        Qwen3TtsModelType.CustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        _ => ModelDownloader.DefaultRepo,
    };

    // ------------------------------------------------------------------------------------------
    // Voice cloning (Base)
    // ------------------------------------------------------------------------------------------

    /// <summary>Builds a voice prompt from 24 kHz mono reference audio. Pass the transcript for ICL mode
    /// (best similarity); with <paramref name="xVectorOnly"/> (or no transcript) only the speaker embedding is used.</summary>
    public VoiceClonePrompt CreateVoiceClonePrompt(float[] referenceAudio24k, string? referenceText, bool xVectorOnly = false)
    {
        Require(Qwen3TtsModelType.Base, "Voice cloning");
        var mel = _mel.Compute(referenceAudio24k, out int frames);
        var spk = _speaker!.Forward(mel, frames, Parallelism);
        bool icl = !xVectorOnly && !string.IsNullOrWhiteSpace(referenceText);
        return new VoiceClonePrompt
        {
            SpeakerEmbedding = spk,
            ReferenceCodes = icl ? _encoder!.Encode(referenceAudio24k, Parallelism) : null,
            ReferenceText = icl ? referenceText : null,
        };
    }

    /// <summary>Loads a WAV reference (any rate / channel count; it is mixed down and resampled to 24 kHz).</summary>
    public VoiceClonePrompt CreateVoiceClonePrompt(string referenceWavPath, string? referenceText, bool xVectorOnly = false)
        => CreateVoiceClonePrompt(AudioFile.ReadMono(referenceWavPath, SampleRate), referenceText, xVectorOnly);

    /// <summary>
    /// Builds a voice prompt from a registered packaged voice (see <see cref="Voices.VoiceRegistry"/>): its Ogg Opus sample
    /// is decoded to 24 kHz and cloned in ICL mode with the sample's exact transcript. The resulting prompt is cached
    /// under <paramref name="cacheDir"/> (default <c>~/.cache/voxwerk/prompts</c>; pass "" to disable), so later uses
    /// skip the speaker and codec encoders.
    /// </summary>
    public VoiceClonePrompt CreateVoiceClonePrompt(string voiceId, string? cacheDir = null)
    {
        Require(Qwen3TtsModelType.Base, "Voice cloning");
        var info = Voices.VoiceRegistry.GetVoiceInfo(voiceId)
                   ?? throw new ArgumentException($"Unknown voice '{voiceId}'. Register a voice package (e.g. EnglishVoices.RegisterAll()) or check `voxwerk voices`.", nameof(voiceId));
        cacheDir ??= Path.Combine(ModelDownloader.DefaultCacheDir, "prompts");
        string? cacheFile = cacheDir.Length == 0 ? null
            : Path.Combine(cacheDir, $"{Config.TtsModelSize}-{Config.HiddenSize}", info.Id + ".vxp");
        if (cacheFile != null && File.Exists(cacheFile))
        {
            try
            {
                using var fs = File.OpenRead(cacheFile);
                var cached = VoiceClonePrompt.Load(fs);
                if (cached.SpeakerEmbedding.Length == Config.HiddenSize && cached.ReferenceText == info.SampleText) return cached;
            }
            catch (Exception e) when (e is IOException or InvalidDataException or EndOfStreamException)
            {
                // Corrupt or stale cache entry: rebuild below.
            }
        }
        var sample = Voices.VoiceRegistry.GetVoiceSample(info.Id) ?? throw new InvalidDataException($"Voice '{info.Id}' has no sample audio.");
        var prompt = CreateVoiceClonePrompt(sample.Audio, info.SampleText);
        if (cacheFile != null)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(cacheFile)!);
                string tmp = cacheFile + "." + Environment.ProcessId + ".tmp";
                using (var fs = File.Create(tmp)) prompt.Save(fs);
                File.Move(tmp, cacheFile, overwrite: true);
            }
            catch (IOException)
            {
                // Caching is best effort.
            }
        }
        return prompt;
    }

    /// <summary>Generates codec frames [frames][16] for <paramref name="text"/> in a cloned voice.</summary>
    public int[][] GenerateCodes(string text, VoiceClonePrompt voice, string language = "auto", GenerationOptions? options = null,
                                 CancellationToken ct = default, Action<int>? onFrame = null)
    {
        Require(Qwen3TtsModelType.Base, "Voice cloning");
        options ??= new GenerationOptions();
        var prompt = _generator.BuildPrompt(text, language, voice, options.NonStreamingMode ?? false, Parallelism);
        return _generator.Generate(prompt, options, Parallelism, ct, onFrame).ToArray();
    }

    /// <summary>Text → speech in a voice cloned from reference audio (Base models).</summary>
    public SynthesisResult Synthesize(string text, VoiceClonePrompt voice, string language = "auto", GenerationOptions? options = null,
                                      CancellationToken ct = default, Action<int>? onFrame = null)
    {
        Require(Qwen3TtsModelType.Base, "Voice cloning");
        options ??= new GenerationOptions();
        return Run(() => _generator.BuildPrompt(text, language, voice, options.NonStreamingMode ?? false, Parallelism),
                   voice.XVectorOnly ? null : voice.ReferenceCodes, options, ct, onFrame);
    }

    // ------------------------------------------------------------------------------------------
    // Voice design (VoiceDesign) and built-in speakers (CustomVoice)
    // ------------------------------------------------------------------------------------------

    /// <summary>Text → speech in a voice created from a natural-language <paramref name="voiceDescription"/>
    /// (timbre, age, gender, emotion, pace, ...). VoiceDesign models only.</summary>
    public SynthesisResult SynthesizeVoiceDesign(string text, string voiceDescription, string language = "auto", GenerationOptions? options = null,
                                                 CancellationToken ct = default, Action<int>? onFrame = null)
    {
        Require(Qwen3TtsModelType.VoiceDesign, "Voice design");
        options ??= new GenerationOptions();
        return Run(() => _generator.BuildPrompt(text, language, null, null, voiceDescription, null, null, options.NonStreamingMode ?? true, Parallelism),
                   null, options, ct, onFrame);
    }

    /// <summary>Text → speech with a built-in <paramref name="speaker"/> (see <see cref="Speakers"/>), optionally
    /// steered by an <paramref name="instruct"/>ion such as "Speak in a whisper" (1.7B only, see
    /// <see cref="SupportsInstruct"/>). CustomVoice models only.</summary>
    public SynthesisResult SynthesizeCustomVoice(string text, string speaker, string language = "auto", string? instruct = null,
                                                 GenerationOptions? options = null, CancellationToken ct = default, Action<int>? onFrame = null)
    {
        Require(Qwen3TtsModelType.CustomVoice, "Built-in speakers");
        options ??= new GenerationOptions();
        string? ins = SupportsInstruct ? instruct : null;
        return Run(() => _generator.BuildPrompt(text, language, null, speaker, ins, null, null, options.NonStreamingMode ?? true, Parallelism),
                   null, options, ct, onFrame);
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

    private SynthesisResult Run(Func<TalkerGenerator.Prompt> buildPrompt, int[][]? context, GenerationOptions options,
                                CancellationToken ct, Action<int>? onFrame)
    {
        var sw = Stopwatch.StartNew();
        var prompt = buildPrompt();
        var tPrompt = sw.Elapsed;
        sw.Restart();
        var codes = _generator.Generate(prompt, options, Parallelism, ct, onFrame).ToArray();
        var tTalker = sw.Elapsed;
        sw.Restart();
        var audio = codes.Length == 0 ? [] : Decode(codes, context, ct);
        var tDecode = sw.Elapsed;
        return new SynthesisResult(audio, SampleRate, codes, tPrompt, tTalker, tDecode);
    }

    public void Dispose()
    {
        _talker.Dispose();
        _speaker?.Dispose();
        _encoder?.Dispose();
        _decoder.Dispose();
    }
}

using System.Diagnostics;
using System.Globalization;
using Voxwerk;
using Voxwerk.Audio;
using Voxwerk.IO;
using Voxwerk.Talker;

return await Cli.RunAsync(args);

internal static class Cli
{
    private const string Usage = """
        voxwerk - Qwen3-TTS (12Hz 0.6B Base) voice-cloning text-to-speech in pure C#

        Usage:
          voxwerk synth --text <text> --ref <ref.wav> [--ref-text <transcript>] [options]
          voxwerk languages [--model <dir|repo>]
          voxwerk download [--model <repo>]

        Input:
          --text <text>            Text to speak (or --text-file <path>)
          --ref <wav>              Reference voice recording (any sample rate, mixed to mono)
          --ref-text <text>        Transcript of the reference (enables ICL mode, best quality)
          --x-vector-only          Clone from the speaker embedding only (no transcript needed)
          --language <name>        auto (default), english, chinese, german, ...

        Output:
          --out <path>             Output WAV (default: out.wav)
          --float32                Write 32-bit float WAV instead of 16-bit PCM

        Model / runtime:
          --model <dir|repo>       Local model directory or Hugging Face repo id
                                   (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base, cached in ~/.cache/voxwerk)
          --threads <n>            Worker threads (default: all cores; 1 = single core)
          --int8                   Store talker weights as int8 (≈1.7× faster generation, approximate)

        Sampling (defaults from generation_config.json):
          --greedy                 Deterministic argmax decoding
          --seed <n>               Random seed
          --temperature <t>        Talker temperature (0.9)       --top-k <k>      (50)
          --top-p <p>              Talker top-p (1.0)              --repetition-penalty <r> (1.05)
          --sub-temperature <t>    Code predictor temperature (0.9) --sub-top-k <k> (50)
          --max-frames <n>         Max generated 12.5 Hz frames (2048)
          --non-streaming          Put the full text in the prompt (non-streaming text layout)
        """;

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Console.WriteLine(Usage);
            return 0;
        }
        try
        {
            var opts = Parse(args.Skip(1).ToArray());
            switch (args[0])
            {
                case "synth":
                    return await SynthAsync(opts);
                case "languages":
                {
                    using var tts = await LoadAsync(opts);
                    Console.WriteLine(string.Join(Environment.NewLine, tts.Languages));
                    return 0;
                }
                case "download":
                {
                    var dir = await ModelDownloader.EnsureAsync(Get(opts, "model") ?? ModelDownloader.DefaultRepo, progress: new Progress<string>(Console.Error.WriteLine));
                    Console.WriteLine(dir);
                    return 0;
                }
                default:
                    Console.Error.WriteLine($"Unknown command '{args[0]}'.\n");
                    Console.Error.WriteLine(Usage);
                    return 2;
            }
        }
        catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 1;
        }
    }

    private static Dictionary<string, string?> Parse(string[] args)
    {
        var flags = new HashSet<string> { "x-vector-only", "float32", "greedy", "non-streaming", "int8" };
        var d = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument '{args[i]}'.");
            string key = args[i][2..];
            if (flags.Contains(key)) d[key] = "true";
            else if (i + 1 < args.Length) d[key] = args[++i];
            else throw new ArgumentException($"Missing value for --{key}.");
        }
        return d;
    }

    private static string? Get(Dictionary<string, string?> o, string key) => o.TryGetValue(key, out var v) ? v : null;
    private static bool Flag(Dictionary<string, string?> o, string key) => o.ContainsKey(key);
    private static float F(Dictionary<string, string?> o, string key, float dflt) => Get(o, key) is { } v ? float.Parse(v, CultureInfo.InvariantCulture) : dflt;
    private static int I(Dictionary<string, string?> o, string key, int dflt) => Get(o, key) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : dflt;

    private static async Task<Qwen3Tts> LoadAsync(Dictionary<string, string?> o)
    {
        var sw = Stopwatch.StartNew();
        var tts = await Qwen3Tts.LoadAsync(Get(o, "model") ?? ModelDownloader.DefaultRepo, I(o, "threads", 0),
            Flag(o, "int8") ? WeightPrecision.Int8 : WeightPrecision.BFloat16, new Progress<string>(Console.Error.WriteLine));
        Console.Error.WriteLine($"Model loaded in {sw.Elapsed.TotalSeconds:F1}s ({tts.Parallelism.MaxThreads} thread(s), {tts.Precision} talker weights).");
        return tts;
    }

    private static async Task<int> SynthAsync(Dictionary<string, string?> o)
    {
        string text = Get(o, "text") ?? (Get(o, "text-file") is { } tf ? await File.ReadAllTextAsync(tf) : throw new ArgumentException("--text or --text-file is required."));
        string refPath = Get(o, "ref") ?? throw new ArgumentException("--ref <reference.wav> is required (the Base model clones a voice).");
        if (!File.Exists(refPath)) throw new FileNotFoundException($"Reference audio not found: {refPath}");
        string? refText = Get(o, "ref-text") ?? (Get(o, "ref-text-file") is { } rf ? await File.ReadAllTextAsync(rf) : null);
        bool xvec = Flag(o, "x-vector-only");
        if (!xvec && string.IsNullOrWhiteSpace(refText))
        {
            Console.Error.WriteLine("note: no --ref-text given, using x-vector-only cloning (pass the transcript for better similarity).");
            xvec = true;
        }

        bool greedy = Flag(o, "greedy");
        var options = new GenerationOptions
        {
            MaxNewTokens = I(o, "max-frames", 2048),
            RepetitionPenalty = F(o, "repetition-penalty", 1.05f),
            NonStreamingMode = Flag(o, "non-streaming"),
            Seed = Get(o, "seed") is { } s ? int.Parse(s, CultureInfo.InvariantCulture) : null,
            Talker = new SamplingOptions { DoSample = !greedy, Temperature = F(o, "temperature", 0.9f), TopK = I(o, "top-k", 50), TopP = F(o, "top-p", 1f) },
            SubTalker = new SamplingOptions { DoSample = !greedy, Temperature = F(o, "sub-temperature", 0.9f), TopK = I(o, "sub-top-k", 50), TopP = F(o, "sub-top-p", 1f) },
        };

        using var tts = await LoadAsync(o);
        var sw = Stopwatch.StartNew();
        var voice = tts.CreateVoiceClonePrompt(refPath, refText, xvec);
        var tVoice = sw.Elapsed;
        Console.Error.WriteLine($"Voice prompt ({(voice.XVectorOnly ? "x-vector" : $"ICL, {voice.ReferenceCodes!.Length} ref frames")}) in {tVoice.TotalSeconds:F2}s");

        var result = tts.Synthesize(text, voice, Get(o, "language") ?? "auto", options,
            onFrame: n => { if (n % 25 == 0) Console.Error.Write($"\r  generated {n} frames ({n / 12.5:F1}s audio)"); });
        Console.Error.WriteLine();
        string outPath = Get(o, "out") ?? "out.wav";
        AudioFile.WriteWav(outPath, result.Audio, result.SampleRate, Flag(o, "float32"));
        Console.Error.WriteLine($"Wrote {outPath}: {result.AudioSeconds:F2}s audio, {result.Codes.Length} frames");
        Console.Error.WriteLine($"  prompt {result.PromptTime.TotalSeconds:F2}s | talker {result.TalkerTime.TotalSeconds:F2}s " +
                                $"({result.TalkerTime.TotalMilliseconds / Math.Max(1, result.Codes.Length):F0} ms/frame) | decoder {result.DecoderTime.TotalSeconds:F2}s");
        double endToEnd = tVoice.TotalSeconds + result.TotalTime.TotalSeconds;
        Console.Error.WriteLine($"  total {endToEnd:F2}s incl. voice prompt, RTF {endToEnd / result.AudioSeconds:F3}");
        return 0;
    }
}

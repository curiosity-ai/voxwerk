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
        voxwerk - Qwen3-TTS (12Hz) text-to-speech in pure C#

        Usage:
          voxwerk synth  --text <text> --ref <ref.wav> [--ref-text <transcript>] [options]   clone a voice (Base models)
          voxwerk synth  --text <text> --voice-id <id> [options]                           speak with a packaged voice
                 (clones its sample with the Base model; without the voice's audio package it uses `design`)
          voxwerk design --text <text> --voice <description> [options]                     describe a voice (VoiceDesign)
          voxwerk design --text <text> --voice-id <id> [options]                           packaged voice from its design prompt
          voxwerk custom --text <text> --speaker <name> [--instruct <style>] [options]     built-in speakers (CustomVoice)
          voxwerk voices [--search <terms>] [--region <r>] [--country <c>] [--category <c>] [--gender <g>] [--age <a>]
                         [--style narration|conversational|characters|commercial] [--json]
          voxwerk voice <id>                                                             show a packaged voice
          voxwerk voice-sample <id> [--out <file.opus|file.wav>]                         export a voice sample
          voxwerk voice-play <id>                                                        play a voice sample
          voxwerk speakers  [--model <dir|repo>]
          voxwerk languages [--model <dir|repo>]
          voxwerk download  [--model <repo>]

        Default models: synth → Qwen/Qwen3-TTS-12Hz-0.6B-Base, design → Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign,
        custom → Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (use --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice for --instruct).

        Input:
          --text <text>            Text to speak (or --text-file <path>)
          --voice <description>    Voice design: natural-language description of the voice (or --voice-file <path>)
          --speaker <name>         Custom voice: built-in speaker (see `voxwerk speakers`)
          --instruct <style>       Custom voice: speaking-style instruction (1.7B CustomVoice only)
          --ref <wav>              Reference voice recording (any sample rate, mixed to mono)
          --voice-id <id>          Packaged voice (see `voxwerk voices`): synth clones its sample instead of --ref,
                                   design speaks its design prompt instead of --voice
          --ref-text <text>        Transcript of the reference (enables ICL mode, best quality)
                                   (or --ref-text-file <path>)
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
          --non-streaming          Put the full text in the prompt (default for design / custom)
          --streaming              Stream the text alongside generation (default for synth)
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
            Voxwerk.Voices.English.Samples.EnglishVoiceSamples.RegisterAll();
            bool positional = args[0] is "voice" or "voice-sample" or "voice-play";
            var opts = positional ? new Dictionary<string, string?>() : Parse(args.Skip(1).ToArray());
            switch (args[0])
            {
                case "synth":
                    return await SynthAsync(opts);
                case "design":
                    return await DesignAsync(opts);
                case "custom":
                    return await CustomAsync(opts);
                case "languages":
                {
                    using var tts = await LoadAsync(opts, ModelDownloader.DefaultRepo);
                    Console.WriteLine(string.Join(Environment.NewLine, tts.Languages));
                    return 0;
                }
                case "voices":
                    return VoiceCommands.List(opts);
                case "voice":
                    return VoiceCommands.Show(args.Length > 1 ? args[1] : null);
                case "voice-sample":
                    return VoiceCommands.Sample(args.Length > 1 ? args[1] : null, Parse(args.Skip(2).ToArray()));
                case "voice-play":
                    return VoiceCommands.Play(args.Length > 1 ? args[1] : null, Parse(args.Skip(2).ToArray()));
                case "speakers":
                {
                    using var tts = await LoadAsync(opts, Qwen3Tts.ExampleRepo(Qwen3TtsModelType.CustomVoice));
                    foreach (var spk in tts.Speakers)
                        Console.WriteLine(tts.Config.SpeakerDialects.TryGetValue(spk, out var d) ? $"{spk} ({d.Replace('_', ' ')})" : spk);
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

    internal static Dictionary<string, string?> Parse(string[] args)
    {
        var flags = new HashSet<string> { "x-vector-only", "float32", "greedy", "non-streaming", "streaming", "int8" };
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

    internal static string? Get(Dictionary<string, string?> o, string key) => o.TryGetValue(key, out var v) ? v : null;
    internal static bool Flag(Dictionary<string, string?> o, string key) => o.ContainsKey(key);
    private static float F(Dictionary<string, string?> o, string key, float dflt) => Get(o, key) is { } v ? float.Parse(v, CultureInfo.InvariantCulture) : dflt;
    internal static int I(Dictionary<string, string?> o, string key, int dflt) => Get(o, key) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : dflt;

    private static async Task<Qwen3Tts> LoadAsync(Dictionary<string, string?> o, string defaultModel)
    {
        var sw = Stopwatch.StartNew();
        var tts = await Qwen3Tts.LoadAsync(Get(o, "model") ?? defaultModel, I(o, "threads", 0),
            Flag(o, "int8") ? WeightPrecision.Int8 : WeightPrecision.BFloat16, new Progress<string>(Console.Error.WriteLine));
        Console.Error.WriteLine($"Model loaded in {sw.Elapsed.TotalSeconds:F1}s ({tts.ModelType} {tts.Config.TtsModelSize}, {tts.Parallelism.MaxThreads} thread(s), {tts.Precision} talker weights).");
        return tts;
    }

    private static async Task<int> SynthAsync(Dictionary<string, string?> o)
    {
        string text = await Text(o);
        string? voiceId = Get(o, "voice-id");
        string? refPath = Get(o, "ref");
        if (voiceId == null && refPath == null)
            throw new ArgumentException("--ref <reference.wav> or --voice-id <id> is required (the Base model clones a voice; see `voxwerk voices`).");
        if (refPath != null && !File.Exists(refPath)) throw new FileNotFoundException($"Reference audio not found: {refPath}");
        if (voiceId != null && !Voxwerk.Voices.VoiceRegistry.Contains(voiceId))
            throw new ArgumentException($"Unknown voice '{voiceId}'. List voices with `voxwerk voices`.");
        if (voiceId != null && !Voxwerk.Voices.VoiceRegistry.HasSample(voiceId))
        {
            Console.Error.WriteLine($"note: the audio package of '{voiceId}' is not installed; speaking it from its design prompt (VoiceDesign).");
            return await DesignAsync(o);
        }
        string? refText = Get(o, "ref-text") ?? (Get(o, "ref-text-file") is { } rf ? await File.ReadAllTextAsync(rf) : null);
        bool xvec = Flag(o, "x-vector-only");
        if (voiceId == null && !xvec && string.IsNullOrWhiteSpace(refText))
        {
            Console.Error.WriteLine("note: no --ref-text given, using x-vector-only cloning (pass the transcript for better similarity).");
            xvec = true;
        }

        var options = Options(o);
        using var tts = await LoadAsync(o, ModelDownloader.DefaultRepo);
        var sw = Stopwatch.StartNew();
        var voice = voiceId != null ? tts.CreateVoiceClonePrompt(voiceId) : tts.CreateVoiceClonePrompt(refPath!, refText, xvec);
        var tVoice = sw.Elapsed;
        Console.Error.WriteLine($"Voice prompt ({(voice.XVectorOnly ? "x-vector" : $"ICL, {voice.ReferenceCodes!.Length} ref frames")}) in {tVoice.TotalSeconds:F2}s");

        var result = tts.Synthesize(text, voice, Get(o, "language") ?? "auto", options, onFrame: Progress);
        return Report(o, result, tVoice);
    }

    private static async Task<int> DesignAsync(Dictionary<string, string?> o)
    {
        string text = await Text(o);
        string voice = Get(o, "voice") ?? Get(o, "instruct") ?? (Get(o, "voice-file") is { } vf ? await File.ReadAllTextAsync(vf)
            : Get(o, "voice-id") is { } id ? (Voxwerk.Voices.VoiceRegistry.GetVoiceInfo(id) ?? throw new ArgumentException($"Unknown voice '{id}'. List voices with `voxwerk voices`.")).DesignPrompt
            : throw new ArgumentException("--voice <description> or --voice-id <id> is required, e.g. --voice \"A calm middle-aged man with a deep, warm voice\"."));
        using var tts = await LoadAsync(o, Qwen3Tts.ExampleRepo(Qwen3TtsModelType.VoiceDesign));
        var result = tts.SynthesizeVoiceDesign(text, voice, Get(o, "language") ?? "auto", Options(o), onFrame: Progress);
        return Report(o, result, TimeSpan.Zero);
    }

    private static async Task<int> CustomAsync(Dictionary<string, string?> o)
    {
        string text = await Text(o);
        using var tts = await LoadAsync(o, Qwen3Tts.ExampleRepo(Qwen3TtsModelType.CustomVoice));
        string speaker = Get(o, "speaker") ?? throw new ArgumentException($"--speaker is required. Available: {string.Join(", ", tts.Speakers)}.");
        string? instruct = Get(o, "instruct");
        if (instruct != null && !tts.SupportsInstruct)
            Console.Error.WriteLine("note: this CustomVoice checkpoint ignores --instruct (only the 1.7B CustomVoice model supports it).");
        var result = tts.SynthesizeCustomVoice(text, speaker, Get(o, "language") ?? "auto", instruct, Options(o), onFrame: Progress);
        return Report(o, result, TimeSpan.Zero);
    }

    private static async Task<string> Text(Dictionary<string, string?> o)
        => Get(o, "text") ?? (Get(o, "text-file") is { } tf ? await File.ReadAllTextAsync(tf) : throw new ArgumentException("--text or --text-file is required."));

    private static GenerationOptions Options(Dictionary<string, string?> o)
    {
        bool greedy = Flag(o, "greedy");
        return new GenerationOptions
        {
            MaxNewTokens = I(o, "max-frames", 2048),
            RepetitionPenalty = F(o, "repetition-penalty", 1.05f),
            NonStreamingMode = Flag(o, "non-streaming") ? true : Flag(o, "streaming") ? false : null,
            Seed = Get(o, "seed") is { } s ? int.Parse(s, CultureInfo.InvariantCulture) : null,
            Talker = new SamplingOptions { DoSample = !greedy, Temperature = F(o, "temperature", 0.9f), TopK = I(o, "top-k", 50), TopP = F(o, "top-p", 1f) },
            SubTalker = new SamplingOptions { DoSample = !greedy, Temperature = F(o, "sub-temperature", 0.9f), TopK = I(o, "sub-top-k", 50), TopP = F(o, "sub-top-p", 1f) },
        };
    }

    private static void Progress(int n)
    {
        if (n % 25 == 0) Console.Error.Write($"\r  generated {n} frames ({n / 12.5:F1}s audio)");
    }

    private static int Report(Dictionary<string, string?> o, SynthesisResult result, TimeSpan tVoice)
    {
        Console.Error.WriteLine();
        string outPath = Get(o, "out") ?? "out.wav";
        AudioFile.WriteWav(outPath, result.Audio, result.SampleRate, Flag(o, "float32"));
        Console.Error.WriteLine($"Wrote {outPath}: {result.AudioSeconds:F2}s audio, {result.Codes.Length} frames");
        Console.Error.WriteLine($"  prompt {result.PromptTime.TotalSeconds:F2}s | talker {result.TalkerTime.TotalSeconds:F2}s " +
                                $"({result.TalkerTime.TotalMilliseconds / Math.Max(1, result.Codes.Length):F0} ms/frame) | decoder {result.DecoderTime.TotalSeconds:F2}s");
        double endToEnd = tVoice.TotalSeconds + result.TotalTime.TotalSeconds;
        Console.Error.WriteLine($"  total {endToEnd:F2}s{(tVoice > TimeSpan.Zero ? " incl. voice prompt" : "")}, RTF {endToEnd / result.AudioSeconds:F3}");
        return 0;
    }
}

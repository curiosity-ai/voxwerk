using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Voxwerk;
using Voxwerk.Audio;
using Voxwerk.Codec;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Voxwerk.Talker;
using Voxwerk.Text;
using Voxwerk.Voices;

// Builds the Voxwerk.Voices.* packages from a voice spec file:
//   generate  - VoiceDesign (1.7B) talker generation, batched and resumable  -> <work>/codes/<id>.codes
//   clone     - Base-model voice cloning of each design sample saying a fixed sentence -> <work>/codes/<id>.clone.codes
//   decode    - speech decoder, parallel over files                          -> <work>/wav/<id>[.clone].wav
//   pack      - Ogg Opus samples + voices.json into src/Voices/<package>/Resources
var o = Args.Parse(args.Skip(1).ToArray());
switch (args.FirstOrDefault())
{
    case "generate": return Commands.Generate(o);
    case "clone": return Commands.Clone(o);
    case "decode": return Commands.Decode(o);
    case "pack": return Commands.Pack(o);
    default:
        Console.Error.WriteLine("usage: voicebuilder generate|decode|pack --spec <spec.json> --work <dir> [--model <dir>] [--out <src/Voices>] [--batch n] [--threads n] [--only <ids.txt>] [--limit n] [--qa <qa.json>]");
        return 2;
}

/// <summary>One voice to build (the spec file is a JSON array of these).</summary>
public sealed record VoiceSpec
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public string Language { get; init; } = "en";
    public required string Country { get; init; }
    public required string Region { get; init; }
    /// <summary>Package suffix after <c>Voxwerk.Voices.English.</c>, e.g. <c>American.Narration.Female</c> (defaults to the region).</summary>
    public string? Package { get; init; }
    public string Style { get; init; } = "";
    public required string Category { get; init; }
    public required string Gender { get; init; }
    public string Age { get; init; } = "";
    public required string Description { get; init; }
    public List<string> Tags { get; init; } = [];
    /// <summary>Instruction given to the VoiceDesign model.</summary>
    public required string DesignPrompt { get; init; }
    /// <summary>What the sample says ("Hi, I'm ... ").</summary>
    public required string Script { get; init; }
    public int Seed { get; init; }
}

internal static class Args
{
    public static Dictionary<string, string> Parse(string[] a)
    {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i + 1 < a.Length; i += 2) d[a[i].TrimStart('-')] = a[i + 1];
        return d;
    }
}

internal static class Commands
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };

    private static List<VoiceSpec> LoadSpec(Dictionary<string, string> o)
    {
        var spec = JsonSerializer.Deserialize<List<VoiceSpec>>(File.ReadAllText(o["spec"]), Json)!;
        if (o.TryGetValue("only", out var only))
        {
            var ids = File.ReadAllLines(only).Select(l => l.Trim()).Where(l => l.Length > 0).ToHashSet();
            spec = spec.Where(s => ids.Contains(s.Id)).ToList();
        }
        if (o.TryGetValue("limit", out var lim)) spec = spec.Take(int.Parse(lim, CultureInfo.InvariantCulture)).ToList();
        return spec;
    }

    public static int Generate(Dictionary<string, string> o)
    {
        string work = o["work"];
        Directory.CreateDirectory(Path.Combine(work, "codes"));
        var spec = LoadSpec(o);
        int seedOffset = o.TryGetValue("seed-offset", out var so) ? int.Parse(so, CultureInfo.InvariantCulture) : 0;
        bool force = o.ContainsKey("force");
        var todo = spec.Where(s => force || !File.Exists(CodesPath(work, s.Id))).ToList();
        Console.Error.WriteLine($"{spec.Count} voices in spec, {todo.Count} to generate");
        if (todo.Count == 0) return 0;

        string dir = o["model"];
        var par = new Parallelism(o.TryGetValue("threads", out var th) ? int.Parse(th, CultureInfo.InvariantCulture) : 0);
        var cfg = Qwen3TtsConfig.Load(Path.Combine(dir, "config.json"));
        if (cfg.TtsModelType != "voice_design") throw new InvalidOperationException("generate needs a VoiceDesign checkpoint.");
        var tok = Qwen2Tokenizer.FromDirectory(dir);
        using var st = new SafeTensorsFile(Path.Combine(dir, "model.safetensors"));
        using var model = new TalkerModel(st, cfg);
        var gen = new TalkerGenerator(model, tok);
        int batch = o.TryGetValue("batch", out var b) ? int.Parse(b, CultureInfo.InvariantCulture) : 16;
        int maxFrames = o.TryGetValue("max-frames", out var mf) ? int.Parse(mf, CultureInfo.InvariantCulture) : 600;

        var sw = Stopwatch.StartNew();
        long frames = 0;
        int done = 0;
        // Prompts are built lazily in chunks so memory stays bounded for thousands of voices.
        foreach (var chunk in todo.Chunk(batch * 8))
        {
            var jobs = chunk.Select(s => new TalkerGenerator.BatchJob(
                gen.BuildPrompt(s.Script, "english", null, null, s.DesignPrompt, null, null, nonStreaming: true, par),
                new GenerationOptions { MaxNewTokens = maxFrames, Seed = s.Seed + seedOffset })).ToList();
            gen.GenerateBatch(jobs, batch, par, (job, codes) =>
            {
                var s = chunk[job];
                WriteCodes(CodesPath(work, s.Id), codes);
                frames += codes.Count;
                done++;
                if (done % 10 == 0 || done == todo.Count)
                    Console.Error.WriteLine($"{done}/{todo.Count} voices, {frames / 12.5 / 60:F1} min audio, {sw.Elapsed.TotalHours:F2} h, {sw.Elapsed.TotalMilliseconds / Math.Max(1, frames):F0} ms/frame");
            });
        }
        return 0;
    }

    /// <summary>The sentence every voice reads in its clone sample.</summary>
    public const string CloneSentence = "Every voice in this collection reads this same sentence, so you can compare how each one sounds side by side.";

    public static int Clone(Dictionary<string, string> o)
    {
        string work = o["work"];
        var spec = LoadSpec(o);
        bool force = o.ContainsKey("force");
        var todo = spec.Where(s => File.Exists(WavPath(work, s.Id)) && (force || !File.Exists(CodesPath(work, s.Id + ".clone")))).ToList();
        Console.Error.WriteLine($"{todo.Count} voices to clone");
        int seedOffset = o.TryGetValue("seed-offset", out var so) ? int.Parse(so, CultureInfo.InvariantCulture) : 0;
        if (todo.Count == 0) return 0;
        int threads = o.TryGetValue("threads", out var th) ? int.Parse(th, CultureInfo.InvariantCulture) : 0;
        using var tts = Qwen3Tts.Load(o["model"], threads);
        if (tts.ModelType != Qwen3TtsModelType.Base) throw new InvalidOperationException("clone needs a Base checkpoint.");
        var gen = tts.Generator;
        int batch = o.TryGetValue("batch", out var b) ? int.Parse(b, CultureInfo.InvariantCulture) : 16;
        var sw = Stopwatch.StartNew();
        int done = 0;
        foreach (var chunk in todo.Chunk(batch * 4))
        {
            // Voice prompts (speaker + codec encoder) are built from the design samples with their exact scripts.
            var jobs = chunk.Select(s =>
            {
                var voice = tts.CreateVoiceClonePrompt(AudioFile.ReadMono(WavPath(work, s.Id), 24000), s.Script);
                WriteCodes(Path.Combine(work, "codes", s.Id + ".clone.ctx"), voice.ReferenceCodes!.ToList());
                return new TalkerGenerator.BatchJob(gen.BuildPrompt(CloneSentence, "english", voice, nonStreaming: false, tts.Parallelism),
                    new GenerationOptions { MaxNewTokens = 400, Seed = s.Seed + 7 + seedOffset });
            }).ToList();
            gen.GenerateBatch(jobs, batch, tts.Parallelism, (job, codes) =>
            {
                WriteCodes(CodesPath(work, chunk[job].Id + ".clone"), codes);
                done++;
                if (done % 10 == 0 || done == todo.Count) Console.Error.WriteLine($"{done}/{todo.Count} cloned, {sw.Elapsed.TotalHours:F2} h");
            });
        }
        return 0;
    }

    public static int Decode(Dictionary<string, string> o)
    {
        string work = o["work"];
        Directory.CreateDirectory(Path.Combine(work, "wav"));
        var files = Directory.GetFiles(Path.Combine(work, "codes"), "*.codes")
            .Where(f => !File.Exists(WavPath(work, Path.GetFileNameWithoutExtension(f)))).ToList();
        Console.Error.WriteLine($"{files.Count} to decode");
        using var st = new SafeTensorsFile(Path.Combine(o["model"], "speech_tokenizer", "model.safetensors"));
        using var dec = new SpeechDecoder(st);
        int threads = o.TryGetValue("threads", out var th) ? int.Parse(th, CultureInfo.InvariantCulture) : Environment.ProcessorCount;
        int done = 0;
        var sw = Stopwatch.StartNew();
        double audio = 0;
        Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = threads }, f =>
        {
            var codes = ReadCodes(f);
            if (codes.Count == 0) return;
            string id = Path.GetFileNameWithoutExtension(f);
            string ctxPath = Path.Combine(work, "codes", id + ".ctx");
            float[] wav;
            if (File.Exists(ctxPath))
            {
                // Clone samples: decode behind the reference codes (acoustic context) and cut them off, like Qwen3Tts.Decode.
                // 25 frames (2 s) of context - the same left context the reference uses between decoder chunks.
                var ctx = ReadCodes(ctxPath);
                if (ctx.Count > SpeechDecoder.LeftContext) ctx = ctx.GetRange(ctx.Count - SpeechDecoder.LeftContext, SpeechDecoder.LeftContext);
                var all = ctx.Concat(codes).ToList();
                var full = dec.Decode(all, Parallelism.Single);
                wav = full[(int)((double)ctx.Count / all.Count * full.Length)..];
            }
            else
            {
                wav = dec.Decode(codes, Parallelism.Single);
            }
            string tmp = WavPath(work, id) + ".tmp";
            AudioFile.WriteWav(tmp, wav, 24000, float32: true);
            File.Move(tmp, WavPath(work, id), overwrite: true);
            lock (files)
            {
                done++;
                audio += wav.Length / 24000.0;
                if (done % 25 == 0) Console.Error.WriteLine($"{done}/{files.Count}, {audio / 60:F1} min audio in {sw.Elapsed.TotalMinutes:F1} min");
            }
        });
        return 0;
    }

    public static int Pack(Dictionary<string, string> o)
    {
        string work = o["work"], outRoot = o.TryGetValue("out", out var or) ? or : "src/Voices";
        var spec = LoadSpec(o);
        HashSet<string>? rejected = null;
        if (o.TryGetValue("qa", out var qaPath))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(qaPath));
            rejected = doc.RootElement.EnumerateObject().Where(p => !p.Value.GetProperty("pass").GetBoolean()).Select(p => p.Name).ToHashSet();
        }
        int bitrate = o.TryGetValue("bitrate", out var br) ? int.Parse(br, CultureInfo.InvariantCulture) : 24000;
        var byPackage = new Dictionary<string, List<VoiceInfo>>();
        int skipped = 0;
        foreach (var s in spec)
        {
            string wavPath = WavPath(work, s.Id);
            if (!File.Exists(wavPath) || rejected?.Contains(s.Id) == true)
            {
                skipped++;
                continue;
            }
            string package = s.Package ?? s.Region;
            string pkgDir = Path.Combine(outRoot, $"Voxwerk.Voices.English.{package}");
            if (!Directory.Exists(pkgDir)) throw new DirectoryNotFoundException(pkgDir);
            var pcm = AudioFile.ReadMono(wavPath, 24000);
            var opus = OggOpus.Encode(pcm, 24000, bitrate, new Dictionary<string, string> { ["title"] = s.Name, ["artist"] = "Voxwerk", ["comment"] = s.Id });
            Directory.CreateDirectory(Path.Combine(pkgDir, "Resources", "samples"));
            File.WriteAllBytes(Path.Combine(pkgDir, "Resources", "samples", s.Id + ".opus"), opus);
            string cloneWav = WavPath(work, s.Id + ".clone");
            double cloneSeconds = 0;
            if (File.Exists(cloneWav) && rejected?.Contains(s.Id + ".clone") != true)
            {
                var clonePcm = AudioFile.ReadMono(cloneWav, 24000);
                cloneSeconds = Math.Round(clonePcm.Length / 24000.0, 2);
                File.WriteAllBytes(Path.Combine(pkgDir, "Resources", "samples", s.Id + ".clone.opus"),
                    OggOpus.Encode(clonePcm, 24000, bitrate, new Dictionary<string, string> { ["title"] = s.Name + " (cloned)", ["artist"] = "Voxwerk", ["comment"] = s.Id }));
            }
            var info = new VoiceInfo
            {
                Id = s.Id, Name = s.Name, Language = s.Language, Country = s.Country, Region = s.Region, Style = s.Style, Category = s.Category,
                Gender = s.Gender, Age = s.Age, Description = s.Description, Tags = s.Tags, SampleText = s.Script,
                SampleSeconds = Math.Round(pcm.Length / 24000.0, 2),
                CloneText = cloneSeconds > 0 ? CloneSentence : "",
                CloneSeconds = cloneSeconds,
            };
            (byPackage.TryGetValue(package, out var l) ? l : byPackage[package] = []).Add(info);
        }
        foreach (var (package, list) in byPackage)
        {
            string path = Path.Combine(outRoot, $"Voxwerk.Voices.English.{package}", "Resources", "voices.json");
            var existing = File.Exists(path) ? JsonSerializer.Deserialize<List<VoiceInfo>>(File.ReadAllText(path), Json) ?? [] : [];
            var merged = existing.Where(e => list.All(n => n.Id != e.Id)).Concat(list).OrderBy(v => v.Id, StringComparer.Ordinal).ToList();
            File.WriteAllText(path, JsonSerializer.Serialize(merged, new JsonSerializerOptions(Json) { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
            Console.Error.WriteLine($"{package}: {list.Count} voices packed ({merged.Count} total)");
        }
        if (skipped > 0) Console.Error.WriteLine($"{skipped} voices skipped (not generated or rejected by QA)");
        return 0;
    }

    private static string CodesPath(string work, string id) => Path.Combine(work, "codes", id + ".codes");
    private static string WavPath(string work, string id) => Path.Combine(work, "wav", id + ".wav");

    private static void WriteCodes(string path, List<int[]> frames)
    {
        string tmp = path + ".tmp";
        using (var w = new BinaryWriter(File.Create(tmp)))
        {
            w.Write(frames.Count);
            foreach (var f in frames)
            {
                w.Write(f.Length);
                foreach (int c in f) w.Write((short)c);
            }
        }
        File.Move(tmp, path, overwrite: true);
    }

    private static List<int[]> ReadCodes(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        int n = r.ReadInt32();
        var frames = new List<int[]>(n);
        for (int i = 0; i < n; i++)
        {
            var f = new int[r.ReadInt32()];
            for (int g = 0; g < f.Length; g++) f[g] = r.ReadInt16();
            frames.Add(f);
        }
        return frames;
    }
}

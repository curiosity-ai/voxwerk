using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Voxwerk.Audio;
using Voxwerk.Voices;

/// <summary>CLI commands for browsing, exporting and playing packaged voices.</summary>
internal static class VoiceCommands
{
    public static int List(Dictionary<string, string?> o)
    {
        var voices = VoiceRegistry.Search(Cli.Get(o, "search"), Cli.Get(o, "category"), Cli.Get(o, "country"),
                                          Cli.Get(o, "region"), Cli.Get(o, "gender"), Cli.Get(o, "age")).ToList();
        int limit = Cli.I(o, "limit", int.MaxValue);
        if (Cli.Flag(o, "json"))
        {
            Console.WriteLine(JsonSerializer.Serialize(voices.Take(limit), new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }
        if (voices.Count == 0)
        {
            Console.Error.WriteLine(VoiceRegistry.All.Count == 0 ? "No voice packages are registered." : "No voices match.");
            return 1;
        }
        foreach (var v in voices.Take(limit))
            Console.WriteLine($"{v.Id,-34} {v.Gender,-7} {v.Age,-12} {v.Category,-24} {Trim(v.Description, 70)}");
        Console.Error.WriteLine($"{voices.Count} voice(s){(voices.Count > limit ? $", showing {limit}" : "")}. Regions: " +
                                string.Join(", ", voices.GroupBy(v => v.Region).OrderByDescending(g => g.Count()).Select(g => $"{g.Key} {g.Count()}")));
        return 0;
    }

    public static int Show(string? id)
    {
        var v = Resolve(id);
        Console.WriteLine($"Id:          {v.Id}");
        Console.WriteLine($"Name:        {v.Name}");
        Console.WriteLine($"Language:    {v.Language}");
        Console.WriteLine($"Region:      {v.Region} ({v.Country})");
        Console.WriteLine($"Category:    {v.Category}");
        Console.WriteLine($"Gender/age:  {v.Gender}, {v.Age}");
        Console.WriteLine($"Tags:        {string.Join(", ", v.Tags)}");
        Console.WriteLine($"Description: {v.Description}");
        Console.WriteLine($"Sample:      \"{v.SampleText}\" ({v.SampleSeconds:F1}s, generated from the description)");
        if (v.CloneSeconds > 0) Console.WriteLine($"Clone:       \"{v.CloneText}\" ({v.CloneSeconds:F1}s, re-created by voice cloning; --kind clone)");
        Console.WriteLine($"Package:     {v.Package}");
        return 0;
    }

    private static VoiceSampleKind Kind(Dictionary<string, string?> o) => (Cli.Get(o, "kind") ?? "design").ToLowerInvariant() switch
    {
        "design" or "description" => VoiceSampleKind.Design,
        "clone" or "cloned" => VoiceSampleKind.Clone,
        var k => throw new ArgumentException($"Unknown --kind '{k}' (use design or clone)."),
    };

    public static int Sample(string? id, Dictionary<string, string?> o)
    {
        var v = Resolve(id);
        var kind = Kind(o);
        string outPath = Cli.Get(o, "out") ?? (kind == VoiceSampleKind.Clone ? v.Id + ".clone.opus" : v.Id + ".opus");
        if (outPath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
        {
            var sample = VoiceRegistry.GetVoiceSample(v.Id, kind) ?? throw new InvalidDataException($"Voice '{v.Id}' has no {kind} sample.");
            AudioFile.WriteWav(outPath, sample.Audio, sample.SampleRate);
        }
        else
        {
            using var s = VoiceRegistry.OpenSampleStream(v.Id, kind) ?? throw new InvalidDataException($"Voice '{v.Id}' has no {kind} sample.");
            using var f = File.Create(outPath);
            s.CopyTo(f);
        }
        Console.Error.WriteLine($"Wrote {outPath}");
        return 0;
    }

    public static int Play(string? id, Dictionary<string, string?> o)
    {
        var v = Resolve(id);
        var sample = VoiceRegistry.GetVoiceSample(v.Id, Kind(o)) ?? throw new InvalidDataException($"Voice '{v.Id}' has no {Kind(o)} sample.");
        string tmp = Path.Combine(Path.GetTempPath(), $"voxwerk-{v.Id}-{Environment.ProcessId}.wav");
        AudioFile.WriteWav(tmp, sample.Audio, sample.SampleRate);
        Console.Error.WriteLine($"{v.Name} ({v.Id}, {sample.Kind}): \"{sample.Text}\"");
        try
        {
            if (OperatingSystem.IsWindows())
            {
                if (!PlaySound(tmp, IntPtr.Zero, SND_FILENAME | SND_SYNC)) throw new InvalidOperationException("PlaySound failed.");
                return 0;
            }
            string[][] players = OperatingSystem.IsMacOS()
                ? [["afplay", tmp]]
                : [["paplay", tmp], ["aplay", "-q", tmp], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp], ["play", "-q", tmp]];
            foreach (var cmd in players)
            {
                try
                {
                    using var p = Process.Start(new ProcessStartInfo(cmd[0], cmd[1..]) { UseShellExecute = false, RedirectStandardError = true });
                    if (p == null) continue;
                    p.WaitForExit();
                    if (p.ExitCode == 0) return 0;
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    // Player not installed: try the next one.
                }
            }
            Console.Error.WriteLine($"No audio player found (tried {string.Join(", ", players.Select(p => p[0]))}); export with `voxwerk voice-sample {v.Id} --out sample.wav`.");
            return 1;
        }
        finally
        {
            try { File.Delete(tmp); } catch (IOException) { }
        }
    }

    private static VoiceInfo Resolve(string? id)
    {
        if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("A voice id is required (see `voxwerk voices`).");
        return VoiceRegistry.GetVoiceInfo(id) ?? throw new ArgumentException($"Unknown voice '{id}'. List voices with `voxwerk voices --search ...`.");
    }

    private static string Trim(string s, int n) => s.Length <= n ? s : s[..(n - 1)] + "…";

    private const uint SND_SYNC = 0x0000, SND_FILENAME = 0x00020000;

    [DllImport("winmm.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool PlaySound(string pszSound, IntPtr hmod, uint fdwSound);
}

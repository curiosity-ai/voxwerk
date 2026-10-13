using System.Reflection;
using System.Text.Json;

namespace Voxwerk.Voices;

/// <summary>A voice catalog whose metadata is the embedded resource <c>voices.json</c> of an assembly.</summary>
public class EmbeddedVoicePack : IVoicePack
{
    private readonly Assembly _assembly;
    private readonly Lazy<Dictionary<string, VoiceInfo>> _voices;
    private readonly Lazy<VoiceInfo[]> _ordered;

    internal static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    public EmbeddedVoicePack(Assembly assembly, string name)
    {
        _assembly = assembly;
        Name = name;
        _voices = new(Load);
        _ordered = new(() => _voices.Value.Values.OrderBy(v => v.Id, StringComparer.Ordinal).ToArray());
    }

    public string Name { get; }

    public IReadOnlyList<VoiceInfo> Voices => _ordered.Value;

    private Dictionary<string, VoiceInfo> Load()
    {
        using var s = _assembly.GetManifestResourceStream("voices.json")
                      ?? throw new InvalidDataException($"{Name}: embedded resource 'voices.json' is missing.");
        var list = JsonSerializer.Deserialize<List<VoiceInfo>>(s, Json) ?? [];
        return list.ToDictionary(v => v.Id, v => v with { Package = Name }, StringComparer.OrdinalIgnoreCase);
    }

    public VoiceInfo? GetVoiceInfo(string id) => _voices.Value.TryGetValue(id, out var v) ? v : null;
}

/// <summary>Reference samples embedded in an assembly as <c>&lt;id&gt;.opus</c> (Ogg Opus) resources.</summary>
public class EmbeddedVoiceSamplePack : IVoiceSamplePack
{
    private readonly Assembly _assembly;
    private readonly Lazy<Dictionary<string, string>> _resources;

    public EmbeddedVoiceSamplePack(Assembly assembly, string name)
    {
        _assembly = assembly;
        Name = name;
        _resources = new(() => assembly.GetManifestResourceNames().Where(r => r.EndsWith(".opus", StringComparison.OrdinalIgnoreCase))
                                       .ToDictionary(r => r[..^5], r => r, StringComparer.OrdinalIgnoreCase));
    }

    public string Name { get; }

    public IReadOnlyCollection<string> VoiceIds => _resources.Value.Keys;

    /// <summary>Resource name of a voice's sample.</summary>
    public static string ResourceName(string id) => id + ".opus";

    public Stream? OpenSampleStream(string id)
        => _resources.Value.TryGetValue(id, out var res) ? _assembly.GetManifestResourceStream(res) : null;
}

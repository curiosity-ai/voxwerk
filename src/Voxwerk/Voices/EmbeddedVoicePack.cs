using System.Reflection;
using System.Text.Json;
using Voxwerk.Audio;

namespace Voxwerk.Voices;

/// <summary>
/// A voice pack whose metadata (<c>voices.json</c>) and samples (<c>&lt;id&gt;.opus</c>, Ogg Opus) are embedded
/// resources of an assembly. Voice packages derive from it and only supply their assembly and name.
/// </summary>
public class EmbeddedVoicePack : IVoicePack
{
    private readonly Assembly _assembly;
    private readonly Lazy<Dictionary<string, VoiceInfo>> _voices;
    private readonly Lazy<Dictionary<string, string>> _resources;

    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    public EmbeddedVoicePack(Assembly assembly, string name)
    {
        _assembly = assembly;
        Name = name;
        _resources = new(() => assembly.GetManifestResourceNames().ToDictionary(r => r, r => r, StringComparer.OrdinalIgnoreCase));
        _voices = new(Load);
    }

    public string Name { get; }

    public IReadOnlyList<VoiceInfo> Voices => _voices.Value.Values.OrderBy(v => v.Id, StringComparer.Ordinal).ToArray();

    private Dictionary<string, VoiceInfo> Load()
    {
        using var s = _assembly.GetManifestResourceStream("voices.json")
                      ?? throw new InvalidDataException($"{Name}: embedded resource 'voices.json' is missing.");
        var list = JsonSerializer.Deserialize<List<VoiceInfo>>(s, Json) ?? [];
        return list.ToDictionary(v => v.Id, v => v with { Package = Name }, StringComparer.OrdinalIgnoreCase);
    }

    public VoiceInfo? GetVoiceInfo(string id) => _voices.Value.TryGetValue(id, out var v) ? v : null;

    /// <summary>Resource name of a sample: <c>&lt;id&gt;.opus</c> (design) or <c>&lt;id&gt;.clone.opus</c>.</summary>
    public static string ResourceName(string id, VoiceSampleKind kind) => kind == VoiceSampleKind.Clone ? id + ".clone.opus" : id + ".opus";

    public Stream? OpenSampleStream(string id, VoiceSampleKind kind = VoiceSampleKind.Design)
    {
        var info = GetVoiceInfo(id);
        return info == null || !_resources.Value.TryGetValue(ResourceName(info.Id, kind), out var res) ? null : _assembly.GetManifestResourceStream(res);
    }

    public VoiceSample? GetVoiceSample(string id, VoiceSampleKind kind = VoiceSampleKind.Design)
    {
        var info = GetVoiceInfo(id);
        using var s = OpenSampleStream(id, kind);
        return info == null || s == null ? null : new VoiceSample(info, kind, OggOpus.Decode(s, 24000), 24000);
    }
}

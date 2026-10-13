using System.Collections.Concurrent;

namespace Voxwerk.Voices;

/// <summary>
/// Global registry of available voices. Voice packages register themselves, e.g.
/// <c>Voxwerk.Voices.English.EnglishVoices.RegisterAll()</c> or <c>EnglishBritishNarrationVoices.Register()</c>;
/// voices are then looked up by id (<c>en-us-marigold-vale</c>) or searched.
/// </summary>
public static class VoiceRegistry
{
    private static readonly ConcurrentDictionary<string, IVoicePack> Packs = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, IVoicePack> ById = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Registers a pack (idempotent). Voice ids already registered by another pack are rejected.</summary>
    public static void Register(IVoicePack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);
        if (!Packs.TryAdd(pack.Name, pack)) return;
        foreach (var v in pack.Voices)
        {
            if (!ById.TryAdd(v.Id, pack) && !ReferenceEquals(ById[v.Id], pack))
                throw new InvalidOperationException($"Voice id '{v.Id}' is provided by both '{ById[v.Id].Name}' and '{pack.Name}'.");
        }
    }

    /// <summary>Registered packs.</summary>
    public static IReadOnlyCollection<IVoicePack> RegisteredPacks => Packs.Values.OrderBy(p => p.Name, StringComparer.Ordinal).ToArray();

    /// <summary>All registered voices, ordered by id.</summary>
    public static IReadOnlyList<VoiceInfo> All => Packs.Values.SelectMany(p => p.Voices).OrderBy(v => v.Id, StringComparer.Ordinal).ToArray();

    public static bool Contains(string id) => ById.ContainsKey(id);

    /// <summary>Voice metadata by id, or null when no registered pack provides it.</summary>
    public static VoiceInfo? GetVoiceInfo(string id) => ById.TryGetValue(id, out var p) ? p.GetVoiceInfo(id) : null;

    /// <summary>Decoded 24 kHz sample by id, or null.</summary>
    public static VoiceSample? GetVoiceSample(string id, VoiceSampleKind kind = VoiceSampleKind.Design)
        => ById.TryGetValue(id, out var p) ? p.GetVoiceSample(id, kind) : null;

    /// <summary>Encoded (Ogg Opus) sample by id, or null.</summary>
    public static Stream? OpenSampleStream(string id, VoiceSampleKind kind = VoiceSampleKind.Design)
        => ById.TryGetValue(id, out var p) ? p.OpenSampleStream(id, kind) : null;

    /// <summary>Searches registered voices. Every non-null filter must match (case-insensitive); <paramref name="query"/>
    /// terms are matched against name, id, description, category, region and tags.</summary>
    public static IEnumerable<VoiceInfo> Search(string? query = null, string? category = null, string? country = null,
                                                string? region = null, string? gender = null, string? age = null, string? language = null,
                                                string? style = null)
    {
        static bool Eq(string? filter, string value) => string.IsNullOrEmpty(filter) || value.Equals(filter, StringComparison.OrdinalIgnoreCase);
        static bool Has(string? filter, string value) => string.IsNullOrEmpty(filter) || value.Contains(filter, StringComparison.OrdinalIgnoreCase);
        return All.Where(v => Has(category, v.Category) && Eq(country, v.Country) && Eq(region, v.Region) && Eq(gender, v.Gender)
                              && Eq(age, v.Age) && Eq(language, v.Language) && Eq(style, v.Style) && v.Matches(query ?? ""));
    }

    /// <summary>Removes every registration (mainly for tests).</summary>
    public static void Clear()
    {
        Packs.Clear();
        ById.Clear();
    }
}

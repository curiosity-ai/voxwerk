using System.Collections.Concurrent;
using Voxwerk.Audio;

namespace Voxwerk.Voices;

/// <summary>
/// Global registry of available voices. A language package registers its catalog (metadata of every voice, enough to
/// speak with the VoiceDesign model), e.g. <c>Voxwerk.Voices.English.EnglishVoices.Register()</c>; optional audio packages
/// add the reference samples needed to clone voices, e.g. <c>EnglishBritishNarrationVoices.Register()</c>. Voices are
/// then looked up by id (<c>en-us-marigold-vale</c>) or searched.
/// </summary>
public static class VoiceRegistry
{
    private static readonly ConcurrentDictionary<string, IVoicePack> Packs = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, IVoicePack> ById = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, IVoiceSamplePack> SamplePacks = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, IVoiceSamplePack> SamplesById = new(StringComparer.OrdinalIgnoreCase);

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

    /// <summary>Registers reference samples (idempotent). The first registered pack wins for a voice id.</summary>
    public static void Register(IVoiceSamplePack samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (!SamplePacks.TryAdd(samples.Name, samples)) return;
        foreach (var id in samples.VoiceIds) SamplesById.TryAdd(id, samples);
    }

    /// <summary>Registered sample packs.</summary>
    public static IReadOnlyCollection<IVoiceSamplePack> RegisteredSamplePacks => SamplePacks.Values.OrderBy(p => p.Name, StringComparer.Ordinal).ToArray();

    /// <summary>Registered packs.</summary>
    public static IReadOnlyCollection<IVoicePack> RegisteredPacks => Packs.Values.OrderBy(p => p.Name, StringComparer.Ordinal).ToArray();

    /// <summary>All registered voices, ordered by id.</summary>
    public static IReadOnlyList<VoiceInfo> All => Packs.Values.SelectMany(p => p.Voices).OrderBy(v => v.Id, StringComparer.Ordinal).ToArray();

    public static bool Contains(string id) => ById.ContainsKey(id);

    /// <summary>Voice metadata by id, or null when no registered pack provides it.</summary>
    public static VoiceInfo? GetVoiceInfo(string id) => ById.TryGetValue(id, out var p) ? p.GetVoiceInfo(id) : null;

    /// <summary>True when a registered sample pack has the reference sample of the voice (so it can be cloned).</summary>
    public static bool HasSample(string id) => SamplesById.ContainsKey(id);

    /// <summary>Encoded (Ogg Opus) reference sample by id, or null when no sample pack for it is registered.</summary>
    public static Stream? OpenSampleStream(string id) => SamplesById.TryGetValue(id, out var p) ? p.OpenSampleStream(id) : null;

    /// <summary>Decoded 24 kHz reference sample by id, or null when the voice or its sample is not registered.</summary>
    public static VoiceSample? GetVoiceSample(string id)
    {
        var info = GetVoiceInfo(id);
        if (info == null) return null;
        using var s = OpenSampleStream(info.Id);
        return s == null ? null : new VoiceSample(info, OggOpus.Decode(s, 24000), 24000);
    }

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
        SamplePacks.Clear();
        SamplesById.Clear();
    }
}

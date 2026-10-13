using Voxwerk.Voices;

namespace Voxwerk.Voices.English.Asian;

/// <summary>Southeast and East Asian Englishes: Filipino, Singaporean, Malaysian, Chinese, Japanese, Korean, Indonesian.</summary>
public static class EnglishAsianVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.Asian";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAsianVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

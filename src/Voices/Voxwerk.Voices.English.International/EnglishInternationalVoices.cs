using Voxwerk.Voices;

namespace Voxwerk.Voices.English.International;

/// <summary>Other and mixed international accents (Caribbean, Middle Eastern, Turkish, unspecified).</summary>
public static class EnglishInternationalVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.International";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishInternationalVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

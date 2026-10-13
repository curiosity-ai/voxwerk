using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Regional;

/// <summary>Regional US accents: Southern, Midwest, Northeast, New York, Boston, Chicago and African American English.</summary>
public static class EnglishAmericanRegionalVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Regional";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAmericanRegionalVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

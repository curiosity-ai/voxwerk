using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Commercial;

/// <summary>General American voices for commercials, radio, news, corporate and announcements.</summary>
public static class EnglishAmericanCommercialVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Commercial";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAmericanCommercialVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

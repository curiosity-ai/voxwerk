using Voxwerk.Voices;

namespace Voxwerk.Voices.English.British.Characters;

/// <summary>British character voices for games, animation and drama.</summary>
public static class EnglishBritishCharactersVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.British.Characters";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishBritishCharactersVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

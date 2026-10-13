using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Narration.Female;

/// <summary>General American female voices for narration: audiobooks, documentaries, e-learning, meditation and storytelling.</summary>
public static class EnglishAmericanNarrationFemaleVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Narration.Female";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAmericanNarrationFemaleVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

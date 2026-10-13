using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Conversational.Male;

/// <summary>General American male voices for conversation: podcasts, assistants, customer service and social media.</summary>
public static class EnglishAmericanConversationalMaleVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Conversational.Male";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAmericanConversationalMaleVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

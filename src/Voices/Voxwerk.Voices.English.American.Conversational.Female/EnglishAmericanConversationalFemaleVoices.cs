using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Conversational.Female;

/// <summary>General American female voices for conversation: podcasts, assistants, customer service and social media.</summary>
public static class EnglishAmericanConversationalFemaleVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Conversational.Female";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishAmericanConversationalFemaleVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

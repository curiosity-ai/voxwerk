using Voxwerk.Voices;

namespace Voxwerk.Voices.English.British.Conversational;

/// <summary>British voices for conversation: podcasts, assistants, customer service and social media.</summary>
public static class EnglishBritishConversationalVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.British.Conversational";

    /// <summary>The voice pack (metadata and Ogg Opus samples embedded in this assembly).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishBritishConversationalVoices).Assembly, PackageName);

    /// <summary>Registers this package's voices with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

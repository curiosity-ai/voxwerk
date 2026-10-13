using Voxwerk.Voices;

namespace Voxwerk.Voices.English.British.Conversational;

/// <summary>Reference samples for cloning: British voices for conversation: podcasts, assistants, customer service and social media.</summary>
/// <remarks>The voice metadata lives in the <c>Voxwerk.Voices.English</c> package; this package only adds the audio needed
/// to clone its voices with the Base model.</remarks>
public static class EnglishBritishConversationalVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.British.Conversational";

    /// <summary>The reference samples (Ogg Opus) embedded in this assembly.</summary>
    public static IVoiceSamplePack Samples { get; } = new EmbeddedVoiceSamplePack(typeof(EnglishBritishConversationalVoices).Assembly, PackageName);

    /// <summary>Registers the English voice catalog and this package's samples with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register()
    {
        Voxwerk.Voices.English.EnglishVoices.Register();
        VoiceRegistry.Register(Samples);
    }
}

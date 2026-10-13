using Voxwerk.Voices;

namespace Voxwerk.Voices.English.American.Narration.Female;

/// <summary>Reference samples for cloning: General American female voices for narration: audiobooks, documentaries, e-learning, meditation and storytelling.</summary>
/// <remarks>The voice metadata lives in the <c>Voxwerk.Voices.English</c> package; this package only adds the audio needed
/// to clone its voices with the Base model.</remarks>
public static class EnglishAmericanNarrationFemaleVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English.American.Narration.Female";

    /// <summary>The reference samples (Ogg Opus) embedded in this assembly.</summary>
    public static IVoiceSamplePack Samples { get; } = new EmbeddedVoiceSamplePack(typeof(EnglishAmericanNarrationFemaleVoices).Assembly, PackageName);

    /// <summary>Registers the English voice catalog and this package's samples with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register()
    {
        Voxwerk.Voices.English.EnglishVoices.Register();
        VoiceRegistry.Register(Samples);
    }
}

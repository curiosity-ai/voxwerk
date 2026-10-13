using Voxwerk.Voices;

namespace Voxwerk.Voices.English;

/// <summary>The English voice catalog: metadata of every English voice (no audio).</summary>
/// <remarks>Every voice can be spoken with the VoiceDesign model from its <see cref="VoiceInfo.DesignPrompt"/>. To clone a
/// voice with the Base model, add the audio package named by <see cref="VoiceInfo.SamplePackage"/> (or
/// <c>Voxwerk.Voices.English.Samples</c> for all of them) and call its <c>Register()</c>.</remarks>
public static class EnglishVoices
{
    /// <summary>NuGet package / pack name.</summary>
    public const string PackageName = "Voxwerk.Voices.English";

    /// <summary>The catalog (embedded <c>voices.json</c>).</summary>
    public static IVoicePack Pack { get; } = new EmbeddedVoicePack(typeof(EnglishVoices).Assembly, PackageName);

    /// <summary>Registers the English voice catalog with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void Register() => VoiceRegistry.Register(Pack);
}

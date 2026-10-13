namespace Voxwerk.Voices;

/// <summary>A set of ready-made voices (metadata + reference audio) that can be registered with <see cref="VoiceRegistry"/>.</summary>
public interface IVoicePack
{
    /// <summary>Package name, e.g. <c>Voxwerk.Voices.English.American.Regional</c>.</summary>
    string Name { get; }

    /// <summary>All voices in the pack.</summary>
    IReadOnlyList<VoiceInfo> Voices { get; }

    /// <summary>Voice metadata by id, or null.</summary>
    VoiceInfo? GetVoiceInfo(string id);

    /// <summary>The encoded sample (Ogg Opus) of a voice, or null.</summary>
    Stream? OpenSampleStream(string id, VoiceSampleKind kind = VoiceSampleKind.Design);

    /// <summary>The decoded 24 kHz sample of a voice, or null.</summary>
    VoiceSample? GetVoiceSample(string id, VoiceSampleKind kind = VoiceSampleKind.Design);
}

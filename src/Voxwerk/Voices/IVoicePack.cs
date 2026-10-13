namespace Voxwerk.Voices;

/// <summary>A catalog of ready-made voices (metadata only) that can be registered with <see cref="VoiceRegistry"/>.
/// Language packages (e.g. <c>Voxwerk.Voices.English</c>) provide one.</summary>
public interface IVoicePack
{
    /// <summary>Package name, e.g. <c>Voxwerk.Voices.English</c>.</summary>
    string Name { get; }

    /// <summary>All voices in the pack.</summary>
    IReadOnlyList<VoiceInfo> Voices { get; }

    /// <summary>Voice metadata by id, or null.</summary>
    VoiceInfo? GetVoiceInfo(string id);
}

/// <summary>Reference samples (Ogg Opus) for some voices of a catalog, needed to clone them. Optional audio packages
/// (e.g. <c>Voxwerk.Voices.English.American.Narration.Female</c>) provide one.</summary>
public interface IVoiceSamplePack
{
    /// <summary>Package name.</summary>
    string Name { get; }

    /// <summary>Ids of the voices this pack has samples for.</summary>
    IReadOnlyCollection<string> VoiceIds { get; }

    /// <summary>The encoded sample (Ogg Opus) of a voice, or null when this pack does not have it.</summary>
    Stream? OpenSampleStream(string id);
}
